"""Quiver daily insider screen → Claude analysis → email.

Screens quiver_signals.db for the day's top clustered-buy insider tickers,
runs Claude AI analysis on each (same logic as quiver_analyze.py), and emails
a ranked report.

Intended cron slot: 7:45 AM Mon-Fri (after quiver pipeline at 7:15 AM).

Usage:
  python quiver_daily.py                    # screen 15, email top 5
  python quiver_daily.py --screen 20 --top 7 --model sonnet
  python quiver_daily.py --no-email         # dry-run, print only
  python quiver_daily.py --min-skin 8.0     # stricter quality filter
"""

from __future__ import annotations

import argparse
import json
import smtplib
import sqlite3
import subprocess
import sys
from datetime import date
from email.mime.text import MIMEText
from pathlib import Path

# Pull analysis functions from the sibling script — avoids duplicating logic.
sys.path.insert(0, str(Path(__file__).parent))
from quiver_analyze import (
    QUIVER_DB, RATING_RANK,
    fetch_quiver, fetch_wsj, fetch_desktop_enrichment,
    build_prompt, call_claude, vote_claude,
)

_DESKTOP_DIR = Path.home() / "gitFinance" / "scanner-desktop"
_ENRICH_PY   = Path(__file__).parent / "quiver_enrich.py"
_ENRICH_VENV = _DESKTOP_DIR / "venv" / "bin" / "python"


def fetch_scanner_enrichment(tickers: list[str]) -> dict[str, dict]:
    """Run scanner-desktop's full enrichment pipeline on all tickers at once.

    Returns a dict keyed by ticker with price, market_cap_B, change_pct,
    zscore, rvol, ou_zscore, conviction_score, conviction_breakdown, and
    all wsj_*/quiver_* enrichment fields. Returns {} on any failure.
    """
    if not (_ENRICH_PY.exists() and _ENRICH_VENV.exists()):
        return {}
    try:
        result = subprocess.run(
            [str(_ENRICH_VENV), str(_ENRICH_PY)] + tickers,
            capture_output=True, text=True,
            cwd=str(_DESKTOP_DIR),
            timeout=300,
        )
        if result.returncode != 0:
            print(f"  [scanner-enrich] WARN: {result.stderr[:300]}")
            return {}
        return json.loads(result.stdout)
    except Exception as exc:
        print(f"  [scanner-enrich] WARN: {exc}")
        return {}

# ── Email credentials (read from env or hard-coded fallback) ─────────────────
import os
SMTP_SERVER   = os.environ.get("QUIVER_SMTP_SERVER",   "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("QUIVER_SMTP_PORT", "587"))
SENDER_EMAIL  = os.environ.get("QUIVER_SENDER_EMAIL",  "alld.illinois@gmail.com")
SENDER_PASS   = os.environ.get("QUIVER_SENDER_PASS",   "vwxbkbpaksutrbdi")
RECIPIENT     = os.environ.get("QUIVER_RECIPIENT",     "nsingh.uiuc@yahoo.com")

# ── Screen query ──────────────────────────────────────────────────────────────
# Ranks by skin-in-game score then cluster count.
# skin_in_game_buy_score weights role (officer > director > 10%-owner) and
# buy size relative to compensation — it's the best single quality filter.
_SCREEN_SQL = """
SELECT symbol,
       cluster_count_buy_30d,
       skin_in_game_buy_score,
       purchase_count_30d,
       ROUND(total_purchase_value / 1e6, 2) AS purch_M,
       has_officer_buy,
       has_director_buy
FROM insider_signals
WHERE score_date = (SELECT MAX(score_date) FROM insider_signals)
  AND is_clustered_buy = 1
  AND skin_in_game_buy_score >= ?
  AND symbol GLOB '[A-Z][A-Z]*'
  AND LENGTH(symbol) BETWEEN 2 AND 5
ORDER BY skin_in_game_buy_score DESC, cluster_count_buy_30d DESC
LIMIT ?
"""


def screen_tickers(n: int, min_skin: float) -> list[dict]:
    with sqlite3.connect(QUIVER_DB) as db:
        rows = db.execute(_SCREEN_SQL, (min_skin, n)).fetchall()
    tickers = []
    for r in rows:
        symbol, clusters, skin, buys, purch_M, officer, director = r
        roles = []
        if officer:
            roles.append("officer")
        if director:
            roles.append("director")
        tickers.append({
            "ticker":       symbol,
            "quiver_bonus": 0.0,
            "price":        0.0,
            "signals":      roles,
            "mktcap_B":     0.0,
            "_clusters":    clusters,
            "_skin":        skin,
            "_purch_M":     purch_M,
        })
    return tickers


# ── Report formatting ───────────────────────────────────────────────────────

def _signal_suffix(r: dict) -> str:
    """Compact 13F-flow + insider-sell annotation — the signals that most often
    explain a Hold (institutions exiting / insiders also selling) but were
    previously invisible in the report."""
    bits = []
    net = r.get("net13_M")
    if net is not None:
        sign = "+" if net >= 0 else ""
        flow = f"13F net {sign}{net}M"
        buyers, sellers = r.get("buyers"), r.get("sellers")
        if buyers is not None and sellers is not None:
            flow += f" ({buyers}b/{sellers}s)"
        bits.append(flow)
    if r.get("sell_flag"):
        ss = r.get("sell_skin")
        bits.append("⚠ insider SELL cluster" + (f" (skin {ss})" if ss else ""))
    return ("  | " + ", ".join(bits)) if bits else ""


# ── Email ─────────────────────────────────────────────────────────────────────

def send_email(subject: str, body: str) -> None:
    msg = MIMEText(body, "plain")
    msg["From"] = SENDER_EMAIL
    msg["To"]   = RECIPIENT
    msg["Subject"] = subject
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASS)
        server.sendmail(SENDER_EMAIL, [RECIPIENT], msg.as_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",     default="haiku",
                   help="Claude model: haiku (default, cheap), sonnet, opus")
    p.add_argument("--screen",    type=int,   default=15,
                   help="Max tickers to pull from the insider screen (default 15)")
    p.add_argument("--top",       type=int,   default=5,
                   help="Top picks to highlight in the email (default 5)")
    p.add_argument("--min-skin",  type=float, default=7.0,
                   help="Minimum skin-in-game score to include (default 7.0)")
    p.add_argument("--votes",     type=int,   default=1, choices=[1, 3],
                   help="1=single call (default), 3=majority vote (2 calls, 3rd on split)")
    p.add_argument("--date",      default=date.today().isoformat(),
                   help="Signal date ceiling YYYY-MM-DD (default today)")
    p.add_argument("--no-email",  action="store_true",
                   help="Print report without sending email")
    args = p.parse_args()

    candidates = screen_tickers(args.screen, args.min_skin)
    if not candidates:
        print(f"No clustered-buy tickers found for {args.date} "
              f"(min skin={args.min_skin})")
        sys.exit(0)

    print(f"Quiver daily  {args.date}  model={args.model}  "
          f"screen={len(candidates)}  min_skin={args.min_skin}\n")

    results: list[tuple] = []
    total_cost   = 0.0
    total_tokens = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}

    # Batch-fetch scanner-desktop enrichment (price, zscore, rvol, conviction, …)
    all_tickers = [c["ticker"] for c in candidates]
    print(f"  Fetching scanner enrichment for {len(all_tickers)} tickers…")
    scanner_data = fetch_scanner_enrichment(all_tickers)
    print(f"  Got enrichment for {len(scanner_data)} tickers\n")

    for c in candidates:
        ticker = c["ticker"]
        enr    = scanner_data.get(ticker, {})
        price    = enr.get("price", 0.0) or 0.0
        mktcap_B = enr.get("market_cap_B", 0.0) or 0.0
        print(f"  {ticker:6}  clusters={c['_clusters']}  skin={c['_skin']}  "
              f"${c['_purch_M']}M", end=" … ", flush=True)
        f13: dict = {}
        ins: dict = {}
        try:
            qctx     = fetch_quiver(ticker, args.date)
            f13      = qctx.get("f13", {}) or {}        # 13F net flow / buyers / sellers
            ins      = qctx.get("insider", {}) or {}    # incl. sell-cluster signal
            wsj      = fetch_wsj(ticker, args.date)
            desktop  = fetch_desktop_enrichment(ticker)
            prompt   = build_prompt(ticker, price, c["quiver_bonus"],
                                    c["signals"], mktcap_B, qctx, wsj, desktop, enr)
            caller = vote_claude if args.votes == 3 else call_claude
            out, cost, tokens = caller(prompt, args.model)
            total_cost += cost
            for k in total_tokens:
                total_tokens[k] += tokens[k]
            rating    = out.get("rating", "Hold")
            reasoning = out.get("reasoning", "")
            print(f"{rating}  (${cost:.3f})")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")
            rating, reasoning = "Hold", f"ERROR: {e}"

        results.append({
            "ticker": ticker, "rating": rating, "reasoning": reasoning,
            "clusters": c["_clusters"], "skin": c["_skin"], "purch_M": c["_purch_M"],
            "price": price, "mcap_B": mktcap_B,
            "net13_M": f13.get("net_M"), "buyers": f13.get("buyers"),
            "sellers": f13.get("sellers"),
            "sell_flag": ins.get("clustered_sell", False),
            "sell_skin": ins.get("sell_skin"),
        })

    results.sort(key=lambda r: RATING_RANK.get(r["rating"], 99))

    # ── Build report ──────────────────────────────────────────────────────────
    actionable = [r for r in results if r["rating"] in ("Buy", "Overweight")]
    sep = "=" * 62
    lines = [
        f"Quiver Insider Picks — {args.date}",
        f"Model: {args.model}   Screened: {len(candidates)}   "
        f"Cost: ${total_cost:.3f}",
    ]

    if actionable:
        lines += [sep, f"BUY / OVERWEIGHT ({len(actionable)} tickers)", sep]
        for i, r in enumerate(actionable, 1):
            price_str = f"  ${r['price']:.2f}" if r["price"] else ""
            mcap_str  = f"  mcap=${r['mcap_B']:.1f}B" if r["mcap_B"] else ""
            lines.append(
                f"\n{i}. {r['ticker']}: {r['rating']}  "
                f"[{r['clusters']} clusters, skin={r['skin']}, ${r['purch_M']}M bought]"
                f"{price_str}{mcap_str}{_signal_suffix(r)}"
            )
            lines.append(f"   {r['reasoning']}")
    else:
        lines += [sep, "No Buy or Overweight ratings today.", sep]

    # Full ranking — now shows the 13F/sell signals AND the per-pick reasoning,
    # so an all-Hold day is interpretable (you can see WHY each is a Hold).
    lines += [
        f"\n{sep}",
        "Full ranking:",
    ]
    for r in results:
        price_str = f"  ${r['price']:.2f}" if r["price"] else ""
        lines.append(
            f"  {r['ticker']:6} {r['rating']:12}  "
            f"clusters={r['clusters']}  skin={r['skin']}  ${r['purch_M']}M{price_str}"
            f"{_signal_suffix(r)}"
        )
        if r["reasoning"]:
            lines.append(f"       {r['reasoning']}")

    lines += [
        "",
        f"Total cost: ${total_cost:.3f}",
        f"Tokens: input={total_tokens['input']:,}  "
        f"cache_write={total_tokens['cache_write']:,}  "
        f"cache_read={total_tokens['cache_read']:,}  "
        f"output={total_tokens['output']:,}",
    ]

    body = "\n".join(lines)
    print(f"\n{body}")

    if args.no_email:
        print("\n[--no-email] skipping send")
        return

    subject = (
        f"Quiver: {len(actionable)} actionable — {args.date}"
        if actionable else
        f"Quiver: no actionable picks — {args.date}"
    )
    send_email(subject, body)
    print(f"\nEmail sent → {RECIPIENT}")


if __name__ == "__main__":
    main()
