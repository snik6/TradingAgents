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
import json
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

SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "rating": {"type": "string", "enum": list(RATING_RANK)},
        "reasoning": {"type": "string"},
    },
    "required": ["rating", "reasoning"],
})


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


# ── prompt ────────────────────────────────────────────────────────────────────

def build_prompt(ticker: str, price: float, quiver_bonus: float,
                 signals: list[str], mktcap_B: float,
                 qctx: dict, wsj: dict) -> str:
    lines = [
        "You are a buy-side equity analyst. Rate this stock using ONLY the signals below.",
        f"Ticker: {ticker}  Price: ${price}  Market cap: ${mktcap_B:.1f}B  "
        f"Quiver bonus: {quiver_bonus:+.2f}",
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

    if not any(k in qctx for k in ("insider", "f13", "dark_pool", "gov")) \
            and not wsj:
        lines.append("NOTE: No signal data found in DB for this ticker.")

    lines += [
        "",
        "Rate: Buy / Overweight / Hold / Underweight / Sell.",
        "Weigh signal coherence heavily — insider+institutional alignment beats one strong signal alone.",
        "Penalise large 13F net outflows even when insider clusters are present.",
        "Consider that a 10x from current market cap requires exceptional, multi-year circumstances.",
        "Respond with your rating and 2-3 sentences of reasoning.",
    ]
    return "\n".join(lines)


# ── claude call ───────────────────────────────────────────────────────────────

def call_claude(prompt: str, model: str) -> dict:
    try:
        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence",
             "--output-format", "json",
             "--model", model,
             "--json-schema", SCHEMA,
             prompt],
            capture_output=True, text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude -p timed out after 10 minutes")
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:400])
    data = json.loads(result.stdout)
    if data.get("is_error"):
        raise RuntimeError(str(data))
    cost = data.get("total_cost_usd", 0)
    usage = data.get("usage", {})
    tokens = {
        "input":          usage.get("input_tokens", 0),
        "cache_write":    usage.get("cache_creation_input_tokens", 0),
        "cache_read":     usage.get("cache_read_input_tokens", 0),
        "output":         usage.get("output_tokens", 0),
    }
    return data["structured_output"], cost, tokens


def vote_claude(prompt: str, model: str) -> tuple[dict, float, dict]:
    """Call claude twice; if ratings agree return immediately (2-call cost).
    If they disagree call a third time as tiebreaker (3-call cost).
    Reasoning is taken from the majority call."""
    out1, cost1, tok1 = call_claude(prompt, model)
    out2, cost2, tok2 = call_claude(prompt, model)

    def _add(a, b):
        return {k: a[k] + b[k] for k in a}

    if out1["rating"] == out2["rating"]:
        total_cost = cost1 + cost2
        total_tok  = _add(tok1, tok2)
        return out1, total_cost, total_tok

    out3, cost3, tok3 = call_claude(prompt, model)
    total_cost = cost1 + cost2 + cost3
    total_tok  = _add(_add(tok1, tok2), tok3)

    votes = [out1["rating"], out2["rating"], out3["rating"]]
    from collections import Counter
    majority_rating = Counter(votes).most_common(1)[0][0]
    winner = next(o for o in [out1, out2, out3] if o["rating"] == majority_rating)
    winner = dict(winner)  # don't mutate original
    winner["reasoning"] += f"  [majority {Counter(votes).most_common(1)[0][1]}/3 votes]"
    return winner, total_cost, total_tok


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
    total_cost = 0.0
    total_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}

    for c in candidates:
        ticker = c["ticker"]
        print(f"  {ticker:6}", end=" ", flush=True)
        try:
            qctx = fetch_quiver(ticker, args.date)
            wsj = fetch_wsj(ticker, args.date)
            prompt = build_prompt(ticker, c["price"], c["quiver_bonus"],
                                  c["signals"], c["mktcap_B"], qctx, wsj)
            out, cost, tokens = call_claude(prompt, args.model)
            total_cost += cost
            for k in total_tokens:
                total_tokens[k] += tokens[k]
            rating = out.get("rating", "Hold")
            reasoning = out.get("reasoning", "")
            print(
                f"→ {rating}  (${cost:.3f} | "
                f"in={tokens['input']} cw={tokens['cache_write']} "
                f"cr={tokens['cache_read']} out={tokens['output']})"
            )
            results.append((ticker, rating, reasoning, c["price"], c["mktcap_B"]))
        except Exception as e:  # noqa: BLE001
            print(f"→ ERROR: {e}")
            results.append((ticker, "Hold", f"ERROR: {e}", c["price"], c["mktcap_B"]))

    results.sort(key=lambda r: RATING_RANK.get(r[1], 99))

    print(f"\n{'=' * 70}")
    print(f"TOP {args.top} PICKS  (model={args.model}, total cost ${total_cost:.3f})")
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
    print(f"\nTotal cost: ${total_cost:.3f}")
    print(
        f"Total tokens: {sum(total_tokens.values()):,}  "
        f"(input={total_tokens['input']:,}  "
        f"cache_write={total_tokens['cache_write']:,}  "
        f"cache_read={total_tokens['cache_read']:,}  "
        f"output={total_tokens['output']:,})"
    )


if __name__ == "__main__":
    main()
