"""Analyze Quiver-screened tickers using Claude via claude -p (no API key needed).

Pulls Quiver signals (insider, 13F, dark pool, gov contracts) + WSJ context
per ticker, then calls claude -p with a structured prompt to get a 5-tier rating.

Usage:
  # Pass tickers directly
  python quiver_analyze.py TKO MRP ADC FCN TXT

  # Pipe from quiver screen output (parses "🔵 TICKER | Quiver bonus +0.70 | Price: $27.67" lines)
  cat quiver_screen.txt | python quiver_analyze.py

  # Control model and number of picks
  python quiver_analyze.py --model sonnet --top 5 TKO MRP ADC FCN TXT OPCH BAH
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

QUIVER_DB = Path(
    __import__("os").environ.get("QUIVER_DB",
    str(Path.home() / "gitFinance" / "shared" / "quiver_signals.db"))
)
WSJ_DB = Path(
    __import__("os").environ.get("WSJ_DB",
    str(Path.home() / "gitFinance" / "shared" / "wsj_signals.db"))
)

RATING_RANK = {"Buy": 0, "Overweight": 1, "Hold": 2, "Underweight": 3, "Sell": 4}

# Path to scanner-desktop source — used by fetch_desktop_enrichment()
_DESKTOP_SRC = Path.home() / "gitFinance" / "scanner-desktop" / "src"

RATINGS = list(RATING_RANK)


# ── data fetching ─────────────────────────────────────────────────────────────

def fetch_quiver(ticker: str, date: str) -> dict:
    ctx: dict = {}
    with sqlite3.connect(QUIVER_DB) as db:
        row = db.execute("""
            SELECT cluster_count_buy_30d, purchase_count_30d, total_purchase_value,
                   skin_in_game_buy_score, has_officer_buy, has_director_buy,
                   has_10pct_owner_buy, is_clustered_sell, skin_in_game_sell_score
            FROM insider_signals WHERE symbol=? AND score_date<=?
            ORDER BY score_date DESC LIMIT 1
        """, (ticker, date)).fetchone()
        if row:
            ctx["insider"] = {
                "clusters": row[0], "buys": row[1],
                "purch_M": round((row[2] or 0) / 1e6, 2),
                "skin": row[3],
                "officer": bool(row[4]), "director": bool(row[5]), "owner10": bool(row[6]),
                "clustered_sell": bool(row[7]), "sell_skin": row[8],
            }

        row = db.execute("""
            SELECT new_position_count, exit_count, net_value,
                   buyers_count, sellers_count, top_buyer_fund
            FROM sec13f_signals WHERE symbol=? AND score_date<=?
            ORDER BY score_date DESC LIMIT 1
        """, (ticker, date)).fetchone()
        if row:
            ctx["f13"] = {
                "new_pos": row[0], "exits": row[1],
                "net_M": round((row[2] or 0) / 1e6, 1),
                "buyers": row[3], "sellers": row[4], "top_buyer": row[5],
            }

        row = db.execute("""
            SELECT dpi, otc_total FROM dark_pool_signals
            WHERE symbol=? AND score_date<=?
            ORDER BY score_date DESC LIMIT 1
        """, (ticker, date)).fetchone()
        if row and row[0]:
            ctx["dark_pool"] = {"dpi": round(row[0], 2),
                                "otc_M": round((row[1] or 0) / 1e6, 2)}

        row = db.execute("""
            SELECT total_value_30d, total_value_90d, is_defense_90d,
                   top_agency_90d, surge_ratio_30d_vs_90d
            FROM gov_contract_signals WHERE symbol=? AND score_date<=?
            ORDER BY score_date DESC LIMIT 1
        """, (ticker, date)).fetchone()
        if row and row[0]:
            ctx["gov"] = {
                "val30_M": round((row[0] or 0) / 1e6, 1),
                "val90_M": round((row[1] or 0) / 1e6, 1),
                "defense": bool(row[2]), "top_agency": row[3],
                "surge": round(row[4] or 0, 2),
            }
    return ctx


def fetch_wsj(ticker: str, date: str) -> dict:
    ctx: dict = {}
    with sqlite3.connect(WSJ_DB) as db:
        row = db.execute("""
            SELECT sentiment_momentum, media_zscore_7d, event_pressure_7d
            FROM ticker_rolling_metrics WHERE ticker=? AND date<=?
            ORDER BY date DESC LIMIT 1
        """, (ticker, date)).fetchone()
        if row and any(v is not None for v in row):
            ctx["rolling"] = {
                "sentiment_momentum": row[0],
                "media_zscore_7d": row[1],
                "event_pressure_7d": row[2],
            }

        rows = db.execute("""
            SELECT a.date, a.headline, substr(a.body, 1, 300)
            FROM articles a
            JOIN article_tickers t ON t.article_id = a.id
            WHERE t.ticker=? AND a.date<=?
            ORDER BY a.date DESC LIMIT 3
        """, (ticker, date)).fetchall()
        if rows:
            ctx["articles"] = [
                {"date": r[0], "headline": r[1] or "", "snippet": r[2] or ""}
                for r in rows
            ]
    return ctx


# ── scanner-desktop enrichment ────────────────────────────────────────────────

# Module-level reader cache — loaded once per process, reused per ticker.
_desktop_readers: dict = {}


def fetch_desktop_enrichment(ticker: str) -> dict:
    """
    Enrich a ticker using scanner-desktop's QuiverReader and WSJSignalsReader.

    Returns a flat dict with:
      quiver_bonus        float  — computed conviction bonus (all signal types)
      lobby_30d_M         float  — lobbying spend last 30d ($M)
      lobby_90d_M         float  — lobbying spend last 90d ($M)
      lobby_top_issue     str    — top lobbying issue area
      wsj_macro_label     str    — BULLISH / BEARISH / NEUTRAL
      wsj_macro_score     float
      wsj_newsletter_score float — aggregate newsletter directional score
      wsj_sentiment       float  — today's avg article sentiment
      wsj_net_events      float  — positive minus negative events today
      wsj_pt_change_pct   float  — analyst PT change % (if any)
      wsj_merger_arb      int    — 1 if M&A event detected
      wsj_legal           int    — 1 if legal event detected
      wsj_credit          int    — 1 if credit downgrade detected
      wsj_days_bullish    int    — days with net-positive coverage in 63d window
      wsj_days_bearish    int    — days with net-negative coverage in 63d window
      wsj_narrative_shift_flag int — 1 if abrupt sentiment reversal detected
    """
    global _desktop_readers

    if not _DESKTOP_SRC.exists():
        return {}

    if not _desktop_readers:
        if str(_DESKTOP_SRC) not in sys.path:
            sys.path.insert(0, str(_DESKTOP_SRC))
        try:
            from analysis.quiver.reader import QuiverReader
            from analysis.wsj_signals_reader import WSJSignalsReader

            qr = QuiverReader({
                "enabled": True,
                "db_path": str(QUIVER_DB),
                "modules": {
                    "insider": True, "govcontracts": True,
                    "lobbying": True, "sec13f": True, "darkpool": True,
                },
            })
            qr.load()

            wr = WSJSignalsReader({"enabled": True, "path": str(WSJ_DB)})
            wr.load()

            _desktop_readers["quiver"] = qr
            _desktop_readers["wsj"]    = wr
        except Exception:
            return {}

    qr = _desktop_readers.get("quiver")
    wr = _desktop_readers.get("wsj")
    if not qr or not wr:
        return {}

    candidate = {"symbol": ticker}
    qr.enrich_candidate(candidate)
    wr.enrich_candidate(candidate)
    bonus = qr.score_bonus(candidate)

    lobby_30d = candidate.get("quiver_lobby_value_30d") or 0.0
    lobby_90d = candidate.get("quiver_lobby_value_90d") or 0.0

    return {
        "quiver_bonus":          round(bonus, 2),
        "lobby_30d_M":           round(lobby_30d / 1e6, 2) if lobby_30d else 0.0,
        "lobby_90d_M":           round(lobby_90d / 1e6, 2) if lobby_90d else 0.0,
        "lobby_top_issue":       candidate.get("quiver_lobby_top_issue") or "",
        "wsj_macro_label":       candidate.get("wsj_macro_label") or "NEUTRAL",
        "wsj_macro_score":       candidate.get("wsj_macro_score") or 0.0,
        "wsj_newsletter_score":  candidate.get("wsj_newsletter_score") or 0.0,
        "wsj_sentiment":         candidate.get("wsj_sentiment") or 0.0,
        "wsj_net_events":        candidate.get("wsj_net_events") or 0.0,
        "wsj_pt_change_pct":     candidate.get("wsj_pt_change_pct"),
        "wsj_merger_arb":        candidate.get("wsj_merger_arb_flag") or 0,
        "wsj_legal":             candidate.get("wsj_legal_flag") or 0,
        "wsj_credit":            candidate.get("wsj_credit_flag") or 0,
        "wsj_days_bullish":      candidate.get("wsj_days_bullish") or 0,
        "wsj_days_bearish":      candidate.get("wsj_days_bearish") or 0,
        "wsj_narrative_shift_flag": candidate.get("wsj_narrative_shift_flag") or 0,
    }


# ── prompt ────────────────────────────────────────────────────────────────────

def build_prompt(ticker: str, price: float, quiver_bonus: float,
                 signals: list[str], mktcap_B: float,
                 qctx: dict, wsj: dict, desktop: dict | None = None,
                 scanner: dict | None = None) -> str:
    d = desktop or {}
    s = scanner or {}
    effective_bonus = d.get("quiver_bonus", quiver_bonus)
    eff_price   = s.get("price", price) or price
    eff_mcap    = s.get("market_cap_B", mktcap_B) or mktcap_B
    lines = [
        "You are a buy-side equity analyst. Rate this stock using ONLY the signals below.",
        f"Ticker: {ticker}  Price: ${eff_price}  Market cap: ${eff_mcap:.1f}B  "
        f"Quiver bonus: {effective_bonus:+.2f}",
        f"Signals fired: {', '.join(signals) or 'none'}",
        "",
    ]

    if "insider" in qctx:
        ins = qctx["insider"]
        roles = " ".join(f"[{r}]" for r, v in
                         [("officer", ins["officer"]), ("director", ins["director"]),
                          ("10pct-owner", ins["owner10"])] if v)
        lines.append(
            f"INSIDER (30d): {ins['clusters']} clusters, {ins['buys']} buys, "
            f"${ins['purch_M']}M, skin={ins['skin']}/10 {roles}"
            + (f"  | sell-cluster present (sell skin={ins['sell_skin']})"
               if ins["clustered_sell"] else "")
        )

    if "f13" in qctx:
        f = qctx["f13"]
        sign = "+" if f["net_M"] >= 0 else ""
        lines.append(
            f"13F: {f['new_pos']} new positions, {f['exits']} exits, "
            f"net={sign}${f['net_M']}M  ({f['buyers']} buyers / {f['sellers']} sellers)"
            + (f"  top buyer: {f['top_buyer']}" if f["top_buyer"] else "")
        )

    if "dark_pool" in qctx:
        dp = qctx["dark_pool"]
        lines.append(
            f"DARK POOL: DPI={dp['dpi']} (>0.65 = institutional accumulation), "
            f"OTC volume={dp['otc_M']}M shares"
        )

    if "gov" in qctx:
        g = qctx["gov"]
        lines.append(
            f"GOV CONTRACTS: ${g['val30_M']}M/30d  ${g['val90_M']}M/90d  "
            f"surge={g['surge']}x vs 90d avg  defense={'yes' if g['defense'] else 'no'}  "
            f"top agency: {g['top_agency']}"
        )

    if "rolling" in wsj:
        r = wsj["rolling"]
        lines.append(
            f"WSJ sentiment momentum: {r['sentiment_momentum']}  "
            f"media z-score 7d: {r['media_zscore_7d']}  "
            f"event pressure 7d: {r['event_pressure_7d']}"
        )

    if "articles" in wsj:
        lines.append("WSJ coverage (most recent articles):")
        for a in wsj["articles"]:
            lines.append(f"  [{a['date']}] {a['headline']}: {a['snippet'][:200]}")

    # ── scanner-desktop enrichment ─────────────────────────────────────────
    if d:
        if d.get("lobby_30d_M"):
            lobby = (
                f"LOBBYING: ${d['lobby_30d_M']}M/30d  ${d['lobby_90d_M']}M/90d"
                + (f"  top issue: {d['lobby_top_issue']}" if d.get("lobby_top_issue") else "")
            )
            lines.append(lobby)

        if d.get("wsj_macro_label") and d["wsj_macro_label"] != "NEUTRAL":
            macro = (
                f"MACRO: {d['wsj_macro_label']} ({d['wsj_macro_score']:+.2f})"
            )
            if d.get("wsj_newsletter_score"):
                macro += f"  newsletter score: {d['wsj_newsletter_score']:+.2f}"
            lines.append(macro)

        if d.get("wsj_sentiment") or d.get("wsj_net_events") or d.get("wsj_pt_change_pct"):
            daily = f"WSJ DAILY: sentiment={d.get('wsj_sentiment', 0):.2f}  net_events={d.get('wsj_net_events', 0):+.0f}"
            if d.get("wsj_pt_change_pct") is not None:
                daily += f"  PT_change={d['wsj_pt_change_pct']:+.1f}%"
            lines.append(daily)

        trend_parts = []
        if d.get("wsj_days_bullish") or d.get("wsj_days_bearish"):
            trend_parts.append(f"{d['wsj_days_bullish']}d bullish / {d['wsj_days_bearish']}d bearish (63d window)")
        if d.get("wsj_narrative_shift_flag"):
            trend_parts.append("NARRATIVE SHIFT DETECTED")
        if trend_parts:
            lines.append("WSJ TREND: " + "  ".join(trend_parts))

        flags = [k for k, f in [("M&A-target", d.get("wsj_merger_arb")),
                                  ("legal", d.get("wsj_legal")),
                                  ("credit-downgrade", d.get("wsj_credit"))] if f]
        if flags:
            lines.append(f"WSJ RISK FLAGS: {', '.join(flags)}  — penalise accordingly")

    if s:
        tech_parts = []
        if s.get("zscore") is not None:
            tech_parts.append(f"vol_zscore={s['zscore']}")
        if s.get("rvol") is not None:
            tech_parts.append(f"rvol={s['rvol']}x")
        if s.get("ou_zscore") is not None:
            tech_parts.append(f"ou_z={s['ou_zscore']}")
        if s.get("change_pct") is not None:
            tech_parts.append(f"chg={s['change_pct']:+.2f}%")
        if s.get("hurst_regime"):
            tech_parts.append(f"hurst={s['hurst_regime']}")
        if s.get("gex_regime"):
            tech_parts.append(f"gex={s['gex_regime']}")
        if tech_parts:
            lines.append("TECHNICAL: " + "  ".join(tech_parts))

        if s.get("conviction_score") is not None:
            brk = s.get("conviction_breakdown") or {}
            conv = f"CONVICTION (desktop): score={s['conviction_score']:.1f}"
            brk_parts = []
            for k in ("quiver", "wsj", "berk", "political"):
                if brk.get(k) is not None:
                    brk_parts.append(f"{k}={brk[k]:.2f}")
            if brk_parts:
                conv += "  " + "  ".join(brk_parts)
            lines.append(conv)

    if not any(k in qctx for k in ("insider", "f13", "dark_pool", "gov")) \
            and not wsj and not d:
        lines.append("NOTE: No signal data found in DB for this ticker.")

    lines += [
        "",
        "Rate this stock: Buy / Overweight / Hold / Underweight / Sell.",
        "Weigh signal coherence heavily — insider+institutional alignment beats one strong signal alone.",
        "Penalise large 13F net outflows even when insider clusters are present.",
        "Consider that a 10x from current market cap requires exceptional, multi-year circumstances.",
        "Start your response with exactly one of these words on the first line: Buy, Overweight, Hold, Underweight, Sell.",
        "Then give 2-3 sentences of reasoning.",
    ]
    return "\n".join(lines)


# ── claude call ───────────────────────────────────────────────────────────────

# Resolve the claude binary by full path — under cron, ~/.local/bin is NOT on
# PATH, so a bare "claude" raises FileNotFoundError ([Errno 2]). Fall back to a
# PATH lookup only if the expected location is missing.
_CLAUDE_BIN = Path.home() / ".local" / "bin" / "claude"
_CLAUDE_BIN = str(_CLAUDE_BIN) if _CLAUDE_BIN.exists() else "claude"


def call_claude(prompt: str, model: str) -> tuple[dict, str]:
    import os
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    try:
        result = subprocess.run(
            [_CLAUDE_BIN, "-p", "--no-session-persistence", "--model", model, prompt],
            capture_output=True, text=True,
            timeout=600, env=env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude -p timed out after 10 minutes")
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:400] or f"claude exited {result.returncode}")
    text = result.stdout.strip()
    if not text:
        raise RuntimeError("claude returned empty response")
    lines = [l for l in text.splitlines() if l.strip()]
    first = lines[0].strip().rstrip(".:,") if lines else ""
    rating = first if first in RATINGS else "Hold"
    reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
    return {"rating": rating, "reasoning": reasoning}


def vote_claude(prompt: str, model: str) -> dict:
    """Call claude twice; if ratings agree return immediately.
    If they disagree call a third time as tiebreaker.
    Reasoning is taken from the majority call."""
    from collections import Counter
    out1 = call_claude(prompt, model)
    out2 = call_claude(prompt, model)
    if out1["rating"] == out2["rating"]:
        return out1
    out3 = call_claude(prompt, model)
    votes = [out1["rating"], out2["rating"], out3["rating"]]
    counts = Counter(votes)
    majority_rating, majority_count = counts.most_common(1)[0]
    winner = dict(next(o for o in [out1, out2, out3] if o["rating"] == majority_rating))
    if majority_count >= 2:
        winner["reasoning"] += f"  [majority {majority_count}/3 votes]"
    else:
        winner["reasoning"] += "  [3-way split — tiebreaker]"
    return winner


# ── stdin parser ──────────────────────────────────────────────────────────────

def parse_quiver_stdin(text: str) -> list[dict]:
    """Parse lines like '🔵 TICKER | Quiver bonus +0.70 | Price: $27.67'."""
    candidates: list[dict] = []
    ticker_re = re.compile(
        r'(?:🔵\s*)?([A-Z]{2,5})\s*\|\s*Quiver bonus\s*([+-]?\d+\.\d+)\s*\|\s*Price:\s*\$([0-9.]+)'
    )
    mktcap_re = re.compile(r'Market Cap:\s*\$([0-9.]+)([BM])')
    signals_re = re.compile(r'Signals:\s*(.+)')

    current: dict = {}
    for line in text.splitlines():
        m = ticker_re.search(line)
        if m:
            if current.get("ticker"):
                candidates.append(current)
            current = {
                "ticker": m.group(1),
                "quiver_bonus": float(m.group(2)),
                "price": float(m.group(3)),
                "signals": [],
                "mktcap_B": 0.0,
            }
        if current:
            m = signals_re.search(line)
            if m:
                current["signals"] = [s.strip() for s in m.group(1).split(",")]
            m = mktcap_re.search(line)
            if m:
                v = float(m.group(1))
                current["mktcap_B"] = v if m.group(2) == "B" else v / 1000

    if current.get("ticker"):
        candidates.append(current)
    return candidates


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*",
                   help="Ticker symbols (omit to read quiver output from stdin)")
    p.add_argument("--model", default="haiku",
                   help="Claude model alias: haiku (default, cheap), sonnet, opus")
    p.add_argument("--top", type=int, default=5, help="Number of top picks to highlight")
    p.add_argument("--date", default="2026-05-26",
                   help="Signal date ceiling (YYYY-MM-DD)")
    args = p.parse_args()

    if args.tickers:
        candidates = [
            {"ticker": t.upper(), "quiver_bonus": 0.0, "price": 0.0,
             "signals": [], "mktcap_B": 0.0}
            for t in args.tickers
        ]
    elif not sys.stdin.isatty():
        candidates = parse_quiver_stdin(sys.stdin.read())
        if not candidates:
            raise SystemExit("stdin parse found no tickers — expected '🔵 TICKER | Quiver bonus ...' lines")
    else:
        p.print_help()
        raise SystemExit(1)

    print(f"Analyzing {len(candidates)} tickers  model={args.model}  date<={args.date}\n")

    results: list[tuple] = []

    for c in candidates:
        ticker = c["ticker"]
        print(f"  {ticker:6}", end=" ", flush=True)
        try:
            qctx = fetch_quiver(ticker, args.date)
            wsj = fetch_wsj(ticker, args.date)
            prompt = build_prompt(ticker, c["price"], c["quiver_bonus"],
                                  c["signals"], c["mktcap_B"], qctx, wsj)
            out = call_claude(prompt, args.model)
            rating = out.get("rating", "Hold")
            reasoning = out.get("reasoning", "")
            print(f"→ {rating}")
            results.append((ticker, rating, reasoning, c["price"], c["mktcap_B"]))
        except Exception as e:  # noqa: BLE001
            print(f"→ ERROR: {e}")
            results.append((ticker, "Hold", f"ERROR: {e}", c["price"], c["mktcap_B"]))

    results.sort(key=lambda r: RATING_RANK.get(r[1], 99))

    print(f"\n{'=' * 70}")
    print(f"TOP {args.top} PICKS  (model={args.model})")
    print(f"{'=' * 70}")
    for i, (ticker, rating, reasoning, price, mcap) in enumerate(results[:args.top], 1):
        meta = "  ".join(filter(None, [
            f"${price}" if price else "",
            f"MCap ${mcap:.1f}B" if mcap else "",
        ]))
        print(f"\n{i}. {ticker}: {rating}  {meta}")
        print(f"   {reasoning}")

    print(f"\n{'=' * 70}")
    print("Full ranking:")
    for ticker, rating, _, _, _ in results:
        print(f"  {ticker}: {rating}")


if __name__ == "__main__":
    main()
