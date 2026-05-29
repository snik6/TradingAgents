from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file (expects ANTHROPIC_API_KEY).
load_dotenv()

# Anthropic Claude — split cheap/fast and deep/strong models across the pipeline
# so analyst summaries stay cheap and only the debate/synthesis pays for Sonnet.
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"
config["deep_think_llm"] = "claude-sonnet-4-6"  # researcher debate, risk review
config["quick_think_llm"] = "claude-haiku-4-5"  # analyst summaries
config["max_debate_rounds"] = 1                 # bump to 2+ for richer debates (more tokens)
# Optional: enable extended thinking for the deep steps (more cost, better reasoning).
# config["anthropic_effort"] = "medium"         # "low" | "medium" | "high"

# Mixed-model adversarial debate — opposing roles run on DIFFERENT models so the
# bull/bear and risk debates are genuinely cross-model, not one model arguing with
# itself. Analysts (tool-calling) stay on the llm_provider above (Claude).
#
# PREREQS (all three) — this mapping 401s without them:
#   1. ANTHROPIC_API_KEY in .env — a REAL key. The keyless `claude -p` CLI does
#      NOT work here; LangChain's ChatAnthropic needs an actual API key. The
#      anthropic base provider (analysts + Claude debate roles) needs it too.
#   2. GOOGLE_API_KEY in .env — for the Gemini roles (already set).
#   3. local ollama daemon running with qwen2.5:14b — for the neutral role.
config["debate_models"] = {
    "bull":           {"provider": "google",    "model": "gemini-2.5-pro"},
    "bear":           {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "aggressive":     {"provider": "google",    "model": "gemini-2.5-flash"},
    "conservative":   {"provider": "anthropic", "model": "claude-haiku-4-5"},
    "neutral":        {"provider": "ollama",    "model": "qwen2.5:14b"},
    "research_judge": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    "risk_judge":     {"provider": "anthropic", "model": "claude-sonnet-4-6"},
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
