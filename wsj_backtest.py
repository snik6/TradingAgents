"""Backtest: do WSJ articles precede price moves?  (DB-backed version)

Events come from wsj_signals.db's `article_signals` — LLM-extracted per
(ticker, date), so the ticker attribution and the sentiment are far cleaner
than keyword matching. For each (ticker, date) we measure the stock's forward
return from that day's close over 1/3/5 trading days, minus SPY (alpha).
"""

import sqlite3
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.simplefilter("ignore")

DB = Path.home() / "gitFinance" / "shared" / "wsj_signals.db"

# --- 1. Events: one (date, ticker) per row, with mean LLM sentiment ----------
conn = sqlite3.connect(DB)
events = conn.execute(
    """SELECT date, ticker, AVG(sentiment_score) AS sentiment, COUNT(*) AS n
       FROM article_signals
       WHERE ticker != '__skip__' AND sentiment_score IS NOT NULL
       GROUP BY date, ticker"""
).fetchall()
conn.close()
print(f"{len(events)} (ticker, date) events from article_signals "
      f"({events[0][0]} .. {events[-1][0]})\n")

# --- 2. Prices ---------------------------------------------------------------
tickers = sorted({e[1] for e in events} | {"SPY"})
px = yf.download(tickers, start="2026-04-25", end="2026-06-06",
                 progress=False, auto_adjust=True)["Close"].sort_index()
trading_days = list(px.index)


def fwd_alpha(ticker, date, horizon):
    """Forward return of `ticker` from the close on/after `date` over `horizon`
    trading days, minus SPY over the same window. None if data is short."""
    after = [td for td in trading_days if td >= pd.Timestamp(date)]
    if len(after) < horizon + 1 or ticker not in px.columns:
        return None
    d0, d1 = after[0], after[horizon]
    try:
        s0, s1 = px[ticker].loc[d0], px[ticker].loc[d1]
        m0, m1 = px["SPY"].loc[d0], px["SPY"].loc[d1]
    except KeyError:
        return None
    if any(pd.isna(v) for v in (s0, s1, m0, m1)):
        return None
    return (s1 / s0 - 1) - (m1 / m0 - 1)


# --- 3. Aggregate ------------------------------------------------------------
def stats(rows):
    vals = [r for r in rows if r is not None]
    if not vals:
        return "n=0"
    avg = sum(vals) / len(vals)
    hit = sum(1 for v in vals if v > 0) / len(vals)
    return f"n={len(vals):3d}  avg alpha={avg:+.2%}  hit-rate={hit:.0%}"


for h in (1, 3, 5):
    allr = [fwd_alpha(t, d, h) for d, t, s, n in events]
    posr = [fwd_alpha(t, d, h) for d, t, s, n in events if s is not None and s > 0.15]
    negr = [fwd_alpha(t, d, h) for d, t, s, n in events if s is not None and s < -0.15]
    print(f"+{h}d forward alpha vs SPY:")
    print(f"   ALL WSJ coverage   {stats(allr)}")
    print(f"   POSITIVE sentiment {stats(posr)}")
    print(f"   NEGATIVE sentiment {stats(negr)}")
    print()
