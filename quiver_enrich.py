"""Enrich a list of tickers using the scanner-desktop pipeline.

Runs all DB-only enrichment stages (WSJ, Berkshire, Quiver, political) plus
FMP quote fetch, intraday z-score/rvol, Gamma-OU-HMM analysis, and conviction
scoring. Outputs JSON keyed by ticker.

Called as a subprocess by quiver_daily.py using scanner-desktop's venv —
do not add external dependencies that aren't already in scanner-desktop/venv.

Usage:
  ~/gitFinance/scanner-desktop/venv/bin/python quiver_enrich.py SHAK GEHC SRAD
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# scanner-desktop/src is the module root for all pipeline imports
_SCANNER_SRC = Path.home() / "gitFinance" / "scanner-desktop" / "src"
sys.path.insert(0, str(_SCANNER_SRC))

from config import Config
from utils.fmp_client import create_fmp_client
from pipeline.stages import (
    _stage2_6_political_enrich,
    _stage2_7_wsj_enrich,
    _stage2_8_berkshire_enrich,
    _stage2_9_quiver_enrich,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


# ── JSON serialisation ────────────────────────────────────────────────────────

def _to_json_safe(obj: Any, _depth: int = 0) -> Any:
    """Recursively strip non-serialisable objects, capping depth at 6."""
    if _depth > 6:
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {
            k: _to_json_safe(v, _depth + 1)
            for k, v in obj.items()
            if not k.startswith("_") and not callable(v)
        }
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x, _depth + 1) for x in obj]
    # numpy / pandas scalars
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    try:
        import pandas as pd
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return None  # too large; drop
    except ImportError:
        pass
    return None


# ── Main enrichment ───────────────────────────────────────────────────────────

def enrich(tickers: list[str]) -> dict[str, dict]:
    config = Config()
    fmp    = create_fmp_client(config)

    candidates = [{"symbol": t.upper()} for t in tickers]

    # ── Stage 2.6–2.9: zero-API enrichments (local SQLite reads) ─────────────
    candidates = _stage2_6_political_enrich(candidates, config, logger)
    candidates = _stage2_7_wsj_enrich(candidates, config, logger)
    candidates = _stage2_8_berkshire_enrich(candidates, config, logger)
    candidates = _stage2_9_quiver_enrich(candidates, config, logger)

    # ── FMP quotes: price, market cap, change %, volume ───────────────────────
    for c in candidates:
        sym = c["symbol"]
        try:
            data = fmp.get("quote", params={"symbol": sym})
            if data:
                q = data[0]
                c["price"]       = q.get("price") or 0.0
                c["market_cap"]  = q.get("marketCap") or 0
                c["market_cap_B"] = round((q.get("marketCap") or 0) / 1e9, 2)
                c["change_pct"]  = q.get("changesPercentage") or q.get("changePercentage") or 0.0
                c["volume"]      = q.get("volume") or 0
                c["avg_volume"]  = q.get("avgVolume") or 0
        except Exception as exc:
            logger.warning("quote failed %s: %s", sym, exc)

    # ── Intraday z-score + rvol ───────────────────────────────────────────────
    try:
        from screener.intraday_data import IntradayDataFetcher
        from analysis.zscore_calculator import ZScoreCalculator
        intraday = IntradayDataFetcher(fmp, config)
        zscorer  = ZScoreCalculator(config.get("zscore", {}))
        for c in candidates:
            try:
                vol = intraday.fetch_current_and_historical_volumes(c["symbol"])
                if vol:
                    zscore, _, meta = zscorer.calculate_projected_zscore(
                        vol["current_volume"], vol["historical_volumes"]
                    )
                    c["zscore"] = round(zscore or 0, 2)
                    c["rvol"]   = round(meta.get("rvol") or 0, 2)
            except Exception as exc:
                logger.debug("zscore %s: %s", c["symbol"], exc)
    except Exception as exc:
        logger.warning("ZScore setup failed: %s", exc)

    # ── Gamma-OU-HMM: OU z-score, Hurst, GEX ─────────────────────────────────
    try:
        from analysis.gamma_ou_hmm_scanner import GammaOUHMMScanner
        prices = {c["symbol"]: c.get("price", 0) for c in candidates}
        priced = [c["symbol"] for c in candidates if (c.get("price") or 0) > 0]
        if priced:
            scanner = GammaOUHMMScanner(fmp, config.to_dict())
            scan    = scanner.scan_stocks(priced, prices)
            _by_sym = {s.get("symbol"): s for s in scan.get("stocks", [])}
            for c in candidates:
                s = _by_sym.get(c["symbol"])
                if not s:
                    continue
                ou = s.get("ou_analysis") or {}
                c["ou_zscore"] = round(ou.get("ou_zscore") or 0, 3)
                hurst = ou.get("hurst") or {}
                c["hurst"]       = round(hurst.get("hurst_exponent") or 0, 3)
                c["hurst_regime"] = hurst.get("hurst_regime", "")
                gex = s.get("gex_analysis") or {}
                c["gex_regime"]   = gex.get("volatility_regime", "")
                c["zero_gamma"]   = round(gex.get("zero_gamma_level") or 0, 2)
                c["gamma_score"]  = round(s.get("conviction_score") or 0, 1)
    except Exception as exc:
        logger.warning("GammaOUHMM failed: %s", exc)

    # ── Conviction scoring ────────────────────────────────────────────────────
    # Config.__setitem__ is not supported, so _quiver_reader was never stashed
    # by stage 2.9. Build a plain dict with the reader attached so
    # ConvictionScorer._score_quiver_bonus() can call reader.score_bonus().
    try:
        from analysis.conviction import ConvictionScorer
        from analysis.quiver.reader import QuiverReader
        conviction_cfg = config.to_dict()
        qr = QuiverReader(config.get("quiver", {}))
        qr.load()
        conviction_cfg["_quiver_reader"] = qr
        scorer = ConvictionScorer(conviction_cfg)
        for c in candidates:
            try:
                res = scorer.calculate_score(c)
                c["conviction_score"]     = res.get("score", 0)
                c["conviction_passes"]    = res.get("passes", False)
                c["conviction_breakdown"] = res.get("breakdown", {})
            except Exception as exc:
                logger.debug("conviction %s: %s", c["symbol"], exc)
    except Exception as exc:
        logger.warning("ConvictionScorer setup failed: %s", exc)

    return {c["symbol"]: _to_json_safe(c) for c in candidates}


def main() -> None:
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    elif not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        tickers = json.loads(raw) if raw.startswith("[") else raw.split()
    else:
        print("Usage: python src/quiver_enrich.py TICKER1 TICKER2 ...", file=sys.stderr)
        sys.exit(1)

    results = enrich(tickers)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
