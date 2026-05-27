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
import smtplib
import sqlite3
import sys
from datetime import date
from email.mime.text import MIMEText
from pathlib import Path

# Pull analysis functions from the sibling script — avoids duplicating logic.
sys.path.insert(0, str(Path(__file__).parent))
from quiver_analyze import (
    QUIVER_DB, RATING_RANK,
    fetch_quiver, fetch_wsj, build_prompt, call_claude, vote_claude,
)

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

    for c in candidates:
        ticker = c["ticker"]
        print(f"  {ticker:6}  clusters={c['_clusters']}  skin={c['_skin']}  "
              f"${c['_purch_M']}M", end=" … ", flush=True)
        try:
            qctx   = fetch_quiver(ticker, args.date)
            wsj    = fetch_wsj(ticker, args.date)
            prompt = build_prompt(ticker, c["price"], c["quiver_bonus"],
                                  c["signals"], c["mktcap_B"], qctx, wsj)
            caller = vote_claude if args.votes == 3 else call_claude
            out, cost, tokens = caller(prompt, args.model)
            total_cost += cost
            for k in total_tokens:
                total_tokens[k] += tokens[k]
            rating    = out.get("rating", "Hold")
            reasoning = out.get("reasoning", "")
            print(f"{rating}  (${cost:.3f})")
            results.append((ticker, rating, reasoning,
                            c["_clusters"], c["_skin"], c["_purch_M"]))
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")
            results.append((ticker, "Hold", f"ERROR: {e}",
                            c["_clusters"], c["_skin"], c["_purch_M"]))

    results.sort(key=lambda r: RATING_RANK.get(r[1], 99))

    # ── Build report ──────────────────────────────────────────────────────────
    sep = "=" * 62
    lines = [
        f"Quiver Insider Picks — {args.date}",
        f"Model: {args.model}   Screened: {len(candidates)}   "
        f"Cost: ${total_cost:.3f}",
        sep,
        f"TOP {args.top} PICKS",
        sep,
    ]
    for i, (ticker, rating, reasoning, clusters, skin, purch_M) in \
            enumerate(results[:args.top], 1):
        lines.append(
            f"\n{i}. {ticker}: {rating}  "
            f"[{clusters} clusters, skin={skin}, ${purch_M}M bought]"
        )
        lines.append(f"   {reasoning}")

    lines += [
        f"\n{sep}",
        "Full ranking:",
    ]
    for ticker, rating, _, clusters, skin, purch_M in results:
        lines.append(
            f"  {ticker:6} {rating:12}  "
            f"clusters={clusters}  skin={skin}  ${purch_M}M"
        )

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

    subject = f"Quiver Insider Picks — {args.date}"
    send_email(subject, body)
    print(f"\nEmail sent → {RECIPIENT}")


if __name__ == "__main__":
    main()
