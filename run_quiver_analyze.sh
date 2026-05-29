#!/usr/bin/env bash
# Quiver daily insider screen + Claude analysis + email report.
#
# Runs after the Quiver pipeline (7:15 AM) has populated quiver_signals.db.
#
# Cron entry:
#   45 7 * * 1-5  /home/nik/gitFinance/TradingAgents/run_quiver_analyze.sh >> /home/nik/gitFinance/TradingAgents/logs/cron/quiver_analyze.log 2>&1
#
# Usage:
#   ./run_quiver_analyze.sh                  # normal run
#   ./run_quiver_analyze.sh --no-email       # dry-run, no email
#   ./run_quiver_analyze.sh --model sonnet   # use sonnet instead of haiku

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

PYTHON="$REPO_DIR/venv/bin/python"
LOG_DIR="$REPO_DIR/logs/cron"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$LOG_DIR"

log "=== Quiver analyze $(date '+%Y-%m-%d') ==="
"$PYTHON" quiver_daily.py --votes 3 --catalyst "$@"
log "Done."
