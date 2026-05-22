"""WSJ news vendor for TradingAgents — backed by shared/wsj_signals.db.

The DB is built daily by the scanner-politics pipeline from the WSJ PDF:
full article bodies (`articles`), ticker tags (`article_tickers`), LLM-extracted
events (`article_signals`), per-ticker daily aggregates (`ticker_daily_metrics`)
and a macro scoring row per day (`global_signals`).

This vendor exposes that to the News Analyst:
  * get_news_wsj         — articles + extracted signals for a specific ticker
  * get_global_news_wsj  — the daily macro digest + top-importance articles

No network calls. DB location overridable via the WSJ_DB env var. Every
function returns an informative string and never raises, so vendor routing
stays predictable.
"""

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")

# article_signals event columns -> human label.
_EVENT_LABELS = {
    "event_earnings_beat": "earnings beat", "event_earnings_miss": "earnings miss",
    "event_analyst_upgrade": "analyst upgrade", "event_analyst_downgrade": "analyst downgrade",
    "event_guidance_raise": "guidance raise", "event_guidance_cut": "guidance cut",
    "event_ma_target": "M&A target", "event_ma_acquirer": "M&A acquirer",
    "event_product_launch": "product launch", "event_buyback": "buyback",
    "event_layoff": "layoffs", "event_regulatory": "regulatory action",
    "event_legal": "legal action", "event_dividend_cut": "dividend cut",
    "event_credit_downgrade": "credit downgrade", "event_ceo_departure": "CEO departure",
}


def _db_path() -> Path:
    return Path(
        os.environ.get("WSJ_DB", "~/gitFinance/shared/wsj_signals.db")
    ).expanduser()


def _connect() -> Optional[sqlite3.Connection]:
    path = _db_path()
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _clean(text: str, limit: int) -> str:
    """Collapse whitespace and truncate to limit chars on a word boundary."""
    text = _WS_RE.sub(" ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _headline(row: sqlite3.Row) -> str:
    h = _WS_RE.sub(" ", (row["headline"] or "")).strip()
    return h if len(h) > 8 else _clean(row["body"], 80)


_SUFFIX_RE = re.compile(
    r"\b(inc|corp|corporation|co|ltd|plc|group|holdings|company|comp|the|"
    r"class [a-c]|&)\b\.?", re.IGNORECASE
)


def _company_name_token(ticker: str) -> Optional[str]:
    """Distinctive lowercased company-name token for `ticker` (first ~2 words),
    used to confirm a tagged article is genuinely about the company rather than
    a stray ticker symbol from an embedded quote table. None if unresolved."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        for key in ("shortName", "longName", "displayName"):
            name = info.get(key)
            if not name:
                continue
            core = _SUFFIX_RE.sub("", name.lower())
            words = [w for w in core.split() if len(w) > 1]
            if words:
                return " ".join(words[:2])
    except Exception as e:  # noqa: BLE001 - network/parse failure is non-fatal
        logger.info("WSJ: company-name lookup failed for %s: %s", ticker, e)
    return None


def _signal_summary(conn, ticker: str, start: str, end: str) -> str:
    """One-line digest of extracted events for a ticker over a date range."""
    rows = conn.execute(
        "SELECT * FROM article_signals WHERE ticker=? AND date BETWEEN ? AND ?",
        (ticker, start, end),
    ).fetchall()
    if not rows:
        return ""
    events = []
    for col, label in _EVENT_LABELS.items():
        n = sum(1 for r in rows if r[col])
        if n:
            events.append(f"{label} ×{n}" if n > 1 else label)
    sents = [r["sentiment_score"] for r in rows if r["sentiment_score"] is not None]
    pt = [r["pt_change_pct"] for r in rows if r["pt_change_pct"] is not None]
    parts = []
    if events:
        parts.append("events: " + ", ".join(events))
    if sents:
        parts.append(f"avg sentiment {sum(sents) / len(sents):+.2f}")
    if pt:
        parts.append(f"price-target change {sum(pt) / len(pt):+.1f}%")
    return "  ·  ".join(parts)


def get_news_wsj(ticker: str, start_date: str, end_date: str) -> str:
    """WSJ articles tagged to `ticker` between start_date and end_date,
    plus a summary of extracted signals. Falls back to the macro digest when
    the ticker has no WSJ coverage in range."""
    conn = _connect()
    if conn is None:
        return f"WSJ database not found at {_db_path()} — no WSJ news for {ticker}."
    try:
        rows = conn.execute(
            """SELECT a.date, a.headline, a.body, a.section, a.importance,
                      at.mention_count
               FROM articles a
               JOIN article_tickers at ON a.id = at.article_id
               WHERE at.ticker = ? AND a.date BETWEEN ? AND ?
               ORDER BY a.importance DESC, at.mention_count DESC""",
            (ticker, start_date, end_date),
        ).fetchall()

        # The DB's article bodies are imperfectly segmented and embed quote
        # tables, so a stray ticker symbol can tag an unrelated article. Keep
        # only articles whose body actually contains the company name.
        note = ""
        token = _company_name_token(ticker)
        if token and rows:
            genuine = [r for r in rows if token in (r["body"] or "").lower()]
            if genuine:
                if len(genuine) < len(rows):
                    note = (f"_(filtered {len(rows) - len(genuine)} of {len(rows)} "
                            f"tagged articles that did not mention '{token}')_\n")
                rows = genuine
            else:
                note = (f"_(none of {len(rows)} tagged articles mention '{token}'; "
                        f"tags may be spurious — showing all)_\n")

        if not rows:
            digest = get_global_news_wsj(end_date, _days_between(start_date, end_date), 8)
            return (
                f"## WSJ News for {ticker}, {start_date} to {end_date}:\n\n"
                f"No WSJ articles tagged to {ticker} in this window. "
                f"WSJ macro digest follows as general context:\n\n{digest}"
            )

        out = [f"## WSJ News for {ticker}, {start_date} to {end_date} "
               f"({len(rows)} articles):\n"]
        if note:
            out.append(note)
        sig = _signal_summary(conn, ticker, start_date, end_date)
        if sig:
            out.append(f"**Extracted signals for {ticker}:** {sig}\n")
        for r in rows:
            out.append(f"### {_headline(r)}  ({r['date']}, {r['section']})")
            out.append(_clean(r["body"], 600) + "\n")
        return "\n".join(out)
    except sqlite3.Error as e:
        logger.warning("WSJ DB read failed for %s: %s", ticker, e)
        return f"WSJ database error for {ticker}: {e}"
    finally:
        conn.close()


def get_global_news_wsj(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    """The WSJ daily macro digest ending on curr_date: the pre-scored
    global_signals row plus the top-importance articles of the window."""
    look_back_days = look_back_days if look_back_days else 7
    limit = limit if limit else 10
    conn = _connect()
    if conn is None:
        return f"WSJ database not found at {_db_path()} — no WSJ global news."
    try:
        try:
            start = (datetime.strptime(curr_date, "%Y-%m-%d")
                     - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
        except ValueError as e:
            return f"WSJ: invalid date ({e})"

        out = [f"## WSJ Global Market Digest, {start} to {curr_date}:\n"]

        # Pre-scored macro row (the most recent on or before curr_date).
        g = conn.execute(
            "SELECT * FROM global_signals WHERE date <= ? ORDER BY date DESC LIMIT 1",
            (curr_date,),
        ).fetchone()
        if g:
            out.append(f"**Macro ({g['date']}):** {g['macro_label']} "
                       f"(score {g['macro_score']:+})  ·  Fed: {g['fed_sentiment']}  ·  "
                       f"theme: {g['dominant_theme']}")
            for label, col in (("Risks", "key_risks"),
                               ("Opportunities", "key_opportunities"),
                               ("Themes", "key_themes")):
                items = _json_list(g[col])
                if items:
                    out.append(f"**{label}:** {', '.join(items)}")
            sectors = _json_obj(g["sector_scores"])
            if sectors:
                ranked = sorted(sectors.items(), key=lambda kv: kv[1], reverse=True)
                out.append("**Sector scores:** "
                           + ", ".join(f"{k} {v:+}" for k, v in ranked))
            out.append("")

        # Top-importance articles in the window (skip the staff masthead).
        rows = conn.execute(
            """SELECT date, headline, body, section, importance
               FROM articles
               WHERE date BETWEEN ? AND ?
                 AND body NOT LIKE '%Editor in Chief%'
               ORDER BY importance DESC, word_count DESC
               LIMIT ?""",
            (start, curr_date, limit),
        ).fetchall()
        if rows:
            out.append(f"### Top WSJ articles ({len(rows)}):\n")
            for r in rows:
                out.append(f"- **{_headline(r)}** ({r['date']}) — {_clean(r['body'], 280)}")
        elif not g:
            return f"No WSJ data found between {start} and {curr_date}."
        return "\n".join(out)
    except sqlite3.Error as e:
        logger.warning("WSJ DB read failed for global news: %s", e)
        return f"WSJ database error: {e}"
    finally:
        conn.close()


def _json_list(raw) -> list:
    try:
        v = json.loads(raw) if raw else []
        return [str(x) for x in v] if isinstance(v, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _json_obj(raw) -> dict:
    try:
        v = json.loads(raw) if raw else {}
        return v if isinstance(v, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _days_between(start_date: str, end_date: str) -> int:
    try:
        d0 = datetime.strptime(start_date, "%Y-%m-%d")
        d1 = datetime.strptime(end_date, "%Y-%m-%d")
        return max(1, (d1 - d0).days)
    except ValueError:
        return 7
