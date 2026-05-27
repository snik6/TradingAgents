"""Top-N stock picks: WSJ signal screen -> TradingAgents debate -> ranked output.

Usage:
  python wsj_top_picks.py                                 # anthropic, last full WSJ day
  python wsj_top_picks.py --provider ollama               # free, weaker reasoning
  python wsj_top_picks.py --date 2026-05-22 --screen-n 12 --top 5

WSJ_DB env var overrides the default DB path (~/gitFinance/shared/wsj_signals.db).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import traceback
from pathlib import Path

from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


DEFAULT_DB = Path.home() / "gitFinance" / "shared" / "wsj_signals.db"

# Bullish composite: positive event score, sentiment, sentiment momentum, and
# rising media attention. Excludes credit/legal flags. Regulatory flag is left
# through — it's noisy and often informational.
SCREEN_SQL = """
SELECT
    t.ticker,
    (COALESCE(t.net_event_score, 0)
     + 2.0 * COALESCE(t.avg_sentiment, 0)
     + 1.5 * COALESCE(r.sentiment_momentum, 0)
     + 0.5 * COALESCE(r.media_zscore_7d, 0)) AS composite,
    t.net_event_score, t.avg_sentiment, r.sentiment_momentum, r.media_zscore_7d,
    t.article_count, t.regulatory_flag
FROM ticker_daily_metrics t
LEFT JOIN ticker_rolling_metrics r
       ON t.ticker = r.ticker AND t.date = r.date
WHERE t.date = ?
  AND t.legal_flag = 0
  AND t.credit_flag = 0
  AND (t.article_count > 0 OR r.media_zscore_7d > 1.5)
ORDER BY composite DESC
LIMIT ?;
"""

RATING_RANK = {"Buy": 0, "Overweight": 1, "Hold": 2, "Underweight": 3, "Sell": 4}


def screen(db_path: Path, date: str, n: int) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(SCREEN_SQL, (date, n)).fetchall()


def build_config(provider: str) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if provider == "anthropic":
        cfg["llm_provider"] = "anthropic"
        cfg["deep_think_llm"] = "claude-sonnet-4-6"
        cfg["quick_think_llm"] = "claude-haiku-4-5"
    elif provider == "ollama":
        cfg["llm_provider"] = "ollama"
        cfg["deep_think_llm"] = "mistral-nemo:12b"
        cfg["quick_think_llm"] = "mistral-nemo:12b"
    else:
        raise SystemExit(f"unknown provider: {provider!r}")
    cfg["max_debate_rounds"] = 1
    cfg["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "wsj",
    }
    cfg["tool_vendors"] = {"get_insider_transactions": "yfinance"}
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=("anthropic", "ollama"), default="anthropic")
    p.add_argument("--date", default="2026-05-22",
                   help="trade date (YYYY-MM-DD). Default is the last full WSJ day.")
    p.add_argument("--screen-n", type=int, default=12,
                   help="candidates to screen before running agents")
    p.add_argument("--top", type=int, default=5,
                   help="final picks returned after agent ranking")
    p.add_argument("--db", default=os.getenv("WSJ_DB", str(DEFAULT_DB)))
    args = p.parse_args()

    load_dotenv()

    print(f"[1/2] screening {args.db} for date={args.date} (top {args.screen_n})")
    rows = screen(Path(args.db), args.date, args.screen_n)
    if not rows:
        raise SystemExit(f"no candidates returned by WSJ screen for {args.date}")
    candidates = [r[0] for r in rows]
    print(f"  candidates: {candidates}")
    for ticker, composite, nes, sent, mom, mz, cnt, reg in rows:
        print(f"    {ticker:6} composite={composite:+.2f} "
              f"events={nes or 0:+.1f} sent={sent or 0:+.2f} "
              f"momentum={mom or 0:+.2f} mz7d={mz or 0:+.2f} "
              f"articles={cnt} reg_flag={reg}")

    print(f"\n[2/2] running TradingAgents ({args.provider}) on {len(candidates)} candidates")
    ta = TradingAgentsGraph(debug=True, config=build_config(args.provider))
    decisions: list[tuple[str, str]] = []
    for ticker in candidates:
        print(f"\n{'=' * 70}\n  {ticker} @ {args.date}\n{'=' * 70}", flush=True)
        try:
            _, rating = ta.propagate(ticker, args.date)
            decisions.append((ticker, rating))
            print(f"  >>> {ticker}: {rating}", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad ticker must not abort the batch
            print(f"  ERROR on {ticker}: {e}", flush=True)
            traceback.print_exc()
            decisions.append((ticker, f"ERROR: {e}"))

    decisions.sort(key=lambda d: RATING_RANK.get(d[1], 99))

    print(f"\n{'=' * 70}\nTOP {args.top} PICKS @ {args.date} ({args.provider})\n{'=' * 70}")
    for i, (ticker, rating) in enumerate(decisions[: args.top], 1):
        print(f"  {i}. {ticker}: {rating}")
    print("\nFull ranking:")
    for ticker, rating in decisions:
        print(f"  {ticker}: {rating}")


if __name__ == "__main__":
    main()
