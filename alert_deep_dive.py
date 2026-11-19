"""Deep-dive driver for scanner-desktop's alert_review (stage 2).

Runs the full TradingAgents graph (analysts -> bull/bear debate -> trader ->
risk debate -> portfolio manager) on a few tickers and writes one JSON line per
ticker to --out, so the caller can read partial results if a later ticker
fails or times out.

  venv/bin/python alert_deep_dive.py --tickers PCG,NVAX --date 2026-11-17 \
      --profile local --out /tmp/deep.jsonl --context-file ctx.json

Profiles:
  local  all agents on local Ollama, qwen3:30b-a3b by default (override with
         ALERT_DEEP_OLLAMA_MODEL); needs OLLAMA_CONTEXT_LENGTH >= 16384 on the service
  main   the mixed Gemini Pro + Ollama adversarial config from main.py
"""

import argparse
import json
import os
import traceback

from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

load_dotenv()


def make_config(profile: str) -> dict:
    if profile == "main":
        from main import config  # main.py builds its config at import time (run guarded by __main__)
        return dict(config)
    # Needs OLLAMA_CONTEXT_LENGTH >= 16384 on the ollama service: the 2,048-token
    # default silently truncates TradingAgents' prompts (ITG was analyzed as "Aflac").
    model = os.getenv("ALERT_DEEP_OLLAMA_MODEL", "qwen3:30b-a3b")
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "ollama"
    config["deep_think_llm"] = model
    config["quick_think_llm"] = model
    config["max_debate_rounds"] = 1
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "wsj",
    }
    config["tool_vendors"] = {"get_insider_transactions": "yfinance"}
    return config


def _clip(x, n=3000):
    return x if not isinstance(x, str) or len(x) <= n else x[:n] + " …"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--profile", choices=["local", "main"], default="local")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ta = TradingAgentsGraph(debug=False, config=make_config(args.profile))
    with open(args.out, "a") as out:
        for t in [x.strip().upper() for x in args.tickers.split(",") if x.strip()]:
            rec = {"symbol": t, "date": args.date, "profile": args.profile}
            try:
                state, decision = ta.propagate(t, args.date)
                rec.update({
                    "decision": str(decision),
                    "final_trade_decision": _clip(state.get("final_trade_decision")),
                    "investment_plan": _clip(state.get("investment_plan"), 1500),
                    "trader_plan": _clip(state.get("trader_investment_plan"), 1500),
                })
            except Exception as e:  # noqa: BLE001 - one bad ticker must not abort the rest
                rec["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            out.write(json.dumps(rec, default=str) + "\n")
            out.flush()
            print(f">>> {t}: {rec.get('decision') or rec.get('error')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
