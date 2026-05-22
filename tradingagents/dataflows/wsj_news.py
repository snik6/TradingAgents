"""WSJ plain-text news vendor for TradingAgents.

Reads the ``WSJNewsPaper-{date}_plain.txt`` files produced daily (~6am) by
``scanner-politics/fetch_wsj.py`` and consumed by ``scanner-news``.  This vendor
lets the News Analyst use the same Wall Street Journal "What's News" digest.

The plain text (pdftotext without -layout) uses U+E013 as the bullet character
for What's News items — already-clean 1-2 sentence summaries.  No network calls:
this purely reads local files.  The directory is configurable via the ``WSJ_DIR``
environment variable (default ``~/gitFinance/tmp``).

Parsing logic is adapted from ``scanner-news/src/news_scanner/wsj_fetcher.py``.
"""

import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# U+E013 is the bullet glyph for "What's News" items in the WSJ plain text.
BULLET = ""

# End-of-bullet marker: a page reference ("A2", "B10") or "WSJ.com". Each
# What's News bullet ends with exactly one of these — used to truncate the
# bullet so it does not bleed into the following article or quote tables.
_BULLET_END_RE = re.compile(r"\s(?:[A-Z]\d{1,2}|WSJ\.com)\b")
# Hyphenated line-break: "nar-\nrower" -> "narrower"
_HYPHEN_BREAK_RE = re.compile(r"-\n\s*")
# Collapse runs of spaces/tabs.
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
# Numeric token (price/percent/ticker-table cell), used to detect quote tables.
_NUM_TOKEN_RE = re.compile(r"^[-+]?\$?\d[\d.,]*%?$")

# Quote-table / chart-caption fragments that leak into bullets. "Sym" is the
# WSJ quote-table column header for "Symbol" — its presence (alongside the other
# headers, or a chart data-source citation) marks a bullet as table noise, not
# prose. Dropping these also prevents a ticker filter matching the stray "Sym".
_TABLE_NOISE_RE = re.compile(
    r"\b(?:Sym\s+Close\s+Chg|Net\s+Sym|Close\s+Chg\s+Stock"
    r"|Source:\s*(?:FactSet|Refinitiv|Dow\s+Jones|Bloomberg))\b",
    re.IGNORECASE,
)

# A real What's News bullet is a 1-2 sentence summary; anything longer means the
# chunk over-captured the article or quote table that follows it.
_MAX_BULLET_LEN = 360


def _wsj_dir() -> Path:
    """Directory holding WSJNewsPaper-*_plain.txt files."""
    return Path(os.environ.get("WSJ_DIR", "~/gitFinance/tmp")).expanduser()


def _plain_file(day: datetime) -> Path:
    return _wsj_dir() / f"WSJNewsPaper-{day.strftime('%Y-%m-%d')}_plain.txt"


def _clean(text: str) -> str:
    """Fix hyphenated line breaks and normalise whitespace."""
    text = _HYPHEN_BREAK_RE.sub("", text)
    text = text.replace("\n", " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def _looks_like_table(text: str) -> bool:
    """True when a blob is mostly numbers — i.e. a stock-quote table, not prose."""
    tokens = text.split()
    if len(tokens) < 6:
        return False
    numeric = sum(1 for t in tokens if _NUM_TOKEN_RE.match(t))
    return numeric / len(tokens) > 0.25


def _parse_whats_news(text: str) -> List[Dict]:
    """Extract bullet items from the What's News section.

    Each bullet is truncated at its page reference ("A2", "WSJ.com", ...) so it
    does not bleed into the article or quote table that follows it.  Bullets are
    WSJ's own editorially-curated digest of the day's most important stories —
    the highest-signal content in the paper for a markets news analyst.
    """
    articles = []
    seen = set()
    for chunk in text.split(BULLET)[1:]:  # chunk[0] is the pre-bullet header
        cleaned = _clean(chunk)
        end = _BULLET_END_RE.search(cleaned)
        # Accept the page-ref cut only if it lands within a plausible bullet
        # length; a far-off match belongs to a later article, not this bullet.
        if end and end.start() <= _MAX_BULLET_LEN:
            cleaned = cleaned[: end.start()].strip()
        else:
            cleaned = cleaned[:_MAX_BULLET_LEN].strip()
        if len(cleaned) < 30 or _looks_like_table(cleaned) or _TABLE_NOISE_RE.search(cleaned):
            continue
        if cleaned[:50] in seen:
            continue
        seen.add(cleaned[:50])
        first_period = cleaned.find(". ")
        title = cleaned[: first_period + 1].strip() if first_period > 20 else cleaned[:80].strip()
        articles.append({"title": title, "description": cleaned, "source": "WSJ What's News"})
    return articles


def _parse_wsj_day(day: datetime) -> List[Dict]:
    """Parse the WSJ What's News digest for a single day. [] if no paper."""
    path = _plain_file(day)
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Could not read WSJ file %s: %s", path, e)
        return []
    return _parse_whats_news(text)


def _format(articles: List[Dict]) -> str:
    out = ""
    for a in articles:
        out += f"### {a['title']} (source: {a['source']})\n{a['description']}\n\n"
    return out


def get_news_wsj(ticker: str, start_date: str, end_date: str) -> str:
    """Retrieve WSJ news relevant to a ticker over a date range.

    WSJ "What's News" is a market-wide digest, so articles are filtered for
    mentions of the ticker symbol or company name.  When nothing matches, the
    full digest is returned as general market context.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        return f"WSJ: invalid date range ({e})"

    all_articles, days_with_paper = [], []
    day = start
    while day <= end:
        day_articles = _parse_wsj_day(day)
        if day_articles:
            days_with_paper.append(day.strftime("%Y-%m-%d"))
            all_articles.extend(day_articles)
        day += timedelta(days=1)

    if not all_articles:
        return (
            f"No WSJ paper found for {ticker} between {start_date} and {end_date} "
            f"(searched {_wsj_dir()})."
        )

    # Build relevance terms: the ticker plus the company name from yfinance.
    terms = {ticker.lower()}
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        for key in ("shortName", "longName", "displayName"):
            name = info.get(key)
            if name:
                # Drop common corporate suffixes for looser matching.
                core = re.sub(
                    r"\b(inc|corp|corporation|co|ltd|plc|group|holdings|the)\b\.?",
                    "",
                    name.lower(),
                ).strip()
                if core:
                    terms.add(core)
    except Exception as e:  # noqa: BLE001 - network/parse failures are non-fatal
        logger.info("WSJ: company-name lookup failed for %s: %s", ticker, e)

    ticker_re = re.compile(rf"\b{re.escape(ticker)}\b", re.IGNORECASE)
    matched = [
        a
        for a in all_articles
        if ticker_re.search(a["title"] + " " + a["description"])
        or any(t in (a["title"] + " " + a["description"]).lower() for t in terms if len(t) > 3)
    ]

    header = f"## WSJ News for {ticker}, from {start_date} to {end_date} (papers: {', '.join(days_with_paper)}):\n\n"
    if matched:
        return header + _format(matched)
    return (
        header
        + f"No WSJ articles mentioned {ticker} directly. "
        "Full WSJ market digest follows as general context:\n\n"
        + _format(all_articles)
    )


def get_global_news_wsj(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    """Retrieve the WSJ "What's News" macro digest ending on curr_date."""
    try:
        curr = datetime.strptime(curr_date, "%Y-%m-%d")
    except ValueError as e:
        return f"WSJ: invalid date ({e})"

    # The LLM may pass these as None explicitly — fall back to defaults.
    look_back_days = look_back_days if look_back_days else 7
    limit = limit if limit else 10

    start = curr - timedelta(days=look_back_days)
    collected, days_with_paper = [], []
    # Newest day first so the limit keeps the most recent news.
    for offset in range(look_back_days + 1):
        day = curr - timedelta(days=offset)
        day_articles = _parse_wsj_day(day)
        if not day_articles:
            continue
        days_with_paper.append(day.strftime("%Y-%m-%d"))
        for a in day_articles:
            collected.append(a)
            if len(collected) >= limit:
                break
        if len(collected) >= limit:
            break

    if not collected:
        return (
            f"No WSJ paper found between {start.strftime('%Y-%m-%d')} and {curr_date} "
            f"(searched {_wsj_dir()})."
        )

    return (
        f"## WSJ Global Market News, from {start.strftime('%Y-%m-%d')} to {curr_date} "
        f"(papers: {', '.join(days_with_paper)}):\n\n" + _format(collected)
    )
