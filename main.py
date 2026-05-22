from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config — local Ollama, no API keys, no spend.
# Endpoint defaults to http://localhost:11434/v1 (set in openai_client.py).
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "ollama"
config["deep_think_llm"] = "mistral-nemo:12b"   # researcher debate, risk review
config["quick_think_llm"] = "mistral-nemo:12b"  # analyst summaries
config["max_debate_rounds"] = 1                 # keep low — local inference is slow

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance
    "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance
    "news_data": "yfinance",                 # Options: alpha_vantage, yfinance
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("SYM", "2026-05-21")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
