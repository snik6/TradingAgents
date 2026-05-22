"""Batch runner — full TradingAgents pipeline over several tickers.

Same config as main.py (local Ollama qwen2.5:14b, WSJ news vendor). Each
ticker's full state is logged to ~/.tradingagents/logs/<TICKER>/.
"""

import traceback

from dotenv import load_dotenv

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()

TICKERS = ["HPE", "TTMI", "NAVN", "KEEL"]
TRADE_DATE = "2026-05-22"

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "qwen2.5:14b"
config["quick_think_llm"] = "qwen2.5:14b"
config["max_debate_rounds"] = 1
config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "wsj",
}
config["tool_vendors"] = {"get_insider_transactions": "yfinance"}

ta = TradingAgentsGraph(debug=True, config=config)

results = {}
for ticker in TICKERS:
    print(f"\n{'=' * 70}\nRUNNING {ticker} @ {TRADE_DATE}\n{'=' * 70}", flush=True)
    try:
        _, decision = ta.propagate(ticker, TRADE_DATE)
        results[ticker] = decision
        print(f"\n>>> {ticker} DECISION: {decision}", flush=True)
    except Exception as e:  # noqa: BLE001 - one bad ticker must not abort the batch
        results[ticker] = f"ERROR: {e}"
        traceback.print_exc()

print(f"\n\n{'=' * 70}\nBATCH SUMMARY\n{'=' * 70}", flush=True)
for ticker, decision in results.items():
    print(f"{ticker}: {decision}", flush=True)
