from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file (expects ANTHROPIC_API_KEY).
load_dotenv()

# Gemini + local — no Anthropic API needed (the Anthropic API account has no
# credit balance; the keyless `claude -p` CLI can't be a LangChain provider).
# Base provider is Google so analysts/tool-calling run on Gemini Flash.
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "google"
config["deep_think_llm"] = "gemini-2.5-pro"     # researcher debate, risk review
config["quick_think_llm"] = "gemini-2.5-flash"  # analyst summaries
config["max_debate_rounds"] = 1                 # bump to 2+ for richer debates (more tokens)

# Mixed-model adversarial debate — opposing roles run on DIFFERENT models so the
# bull/bear and risk debates are genuinely cross-model, not one model arguing with
# itself. This mapping uses ONLY providers available now: Gemini (GOOGLE_API_KEY)
# and local ollama (qwen2.5:14b / llama3.1:8b) — no Anthropic API cost.
# To switch to the Claude heavyweight version later, add API credits at
# console.anthropic.com and set bear/judges back to anthropic/claude-sonnet-4-6.
#
# PREREQS: GOOGLE_API_KEY in .env + local ollama daemon running with the models below.
config["debate_models"] = {
    "bull":           {"provider": "google", "model": "gemini-2.5-pro"},
    "bear":           {"provider": "ollama", "model": "qwen2.5:14b"},
    "aggressive":     {"provider": "google", "model": "gemini-2.5-flash"},
    "conservative":   {"provider": "ollama", "model": "llama3.1:8b"},
    "neutral":        {"provider": "ollama", "model": "qwen2.5:14b"},
    "research_judge": {"provider": "google", "model": "gemini-2.5-pro"},
    "risk_judge":     {"provider": "google", "model": "gemini-2.5-pro"},
}

# Data vendors — yfinance is free; WSJ news comes from shared/wsj_signals.db,
# populated by the scanner-politics pipeline. Override the DB path with WSJ_DB.
config["data_vendors"] = {
    "core_stock_apis": "yfinance",          # Options: alpha_vantage, yfinance
    "technical_indicators": "yfinance",     # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",         # Options: alpha_vantage, yfinance
    "news_data": "wsj",                     # Options: wsj, alpha_vantage, yfinance
}

# WSJ has no insider data — route that one tool to yfinance.
config["tool_vendors"] = {
    "get_insider_transactions": "yfinance",
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("SYM", "2026-05-22")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
