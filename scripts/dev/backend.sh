#!/usr/bin/env bash
# backend.sh — run the RLM Studio API server on its own, so it can be stopped
# and restarted while the UI (scripts/dev/frontend.sh) stays up. That is what
# the frontend manual test plan's §1.1, §1.2 and §5.8 need.
#
# Usage:
#   scripts/dev/backend.sh start      # API on :8000 (RLM_STUDIO_PORT to change)
#   scripts/dev/backend.sh stop
#   scripts/dev/backend.sh restart
#   scripts/dev/backend.sh status
#
# Runs in the background; log: ${TMPDIR:-/tmp}/rlm-studio-dev/backend.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${RLM_STUDIO_PORT:-8000}"
TMP_ROOT="${TMPDIR:-/tmp}"
LOG_DIR="${TMP_ROOT%/}/rlm-studio-dev"
LOG="$LOG_DIR/backend.log"
# Only processes whose command line matches this are ever killed.
OWN_PROCESS_PATTERN="rlmstudio|rlm-studio"
START_TIMEOUT_S=60
STOP_TIMEOUT_S=10

listener_pids() {
  lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

is_ours() {
  ps -o command= -p "$1" 2>/dev/null | grep -Eq "$OWN_PROCESS_PATTERN"
}

status() {
  local pids
  pids="$(listener_pids)"
  if [ -z "$pids" ]; then
    echo "backend: stopped (nothing listening on :$PORT)"
    return 1
  fi
  echo "backend: running on :$PORT (pid $(echo $pids))"
}

start() {
  local pids
  pids="$(listener_pids)"
  if [ -n "$pids" ]; then
    echo "backend: :$PORT is already in use (pid $(echo $pids))." >&2
    echo "         Run '$0 stop', or set RLM_STUDIO_PORT to another port." >&2
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  cd "$REPO_ROOT"
  nohup uv run python -m rlmstudio.server --port "$PORT" >"$LOG" 2>&1 &
  echo "backend: starting on :$PORT (log: $LOG)"
  for _ in $(seq "$START_TIMEOUT_S"); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "backend: up — http://localhost:$PORT/health"
      return 0
    fi
    sleep 1
  done
  echo "backend: did not answer /health within ${START_TIMEOUT_S}s — see $LOG" >&2
  exit 1
}

stop() {
  local pids pid
  pids="$(listener_pids)"
  if [ -z "$pids" ]; then
    echo "backend: not running on :$PORT"
    return 0
  fi
  for pid in $pids; do
    if ! is_ours "$pid"; then
      echo "backend: pid $pid on :$PORT is not an RLM Studio server — not touching it:" >&2
      ps -o pid=,command= -p "$pid" >&2
      exit 1
    fi
  done
  # shellcheck disable=SC2086
  kill $pids
  for _ in $(seq "$STOP_TIMEOUT_S"); do
    [ -z "$(listener_pids)" ] && { echo "backend: stopped"; return 0; }
    sleep 1
  done
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  echo "backend: stopped (forced after ${STOP_TIMEOUT_S}s)"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
