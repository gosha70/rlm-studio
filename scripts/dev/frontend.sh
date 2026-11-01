#!/usr/bin/env bash
# frontend.sh — run the RLM Studio UI as its own process on :3000, proxying
# /api, /health and /ws to the backend (scripts/dev/backend.sh). Stopping one
# leaves the other running.
#
# Usage:
#   scripts/dev/frontend.sh start         # production build + `next start`
#   scripts/dev/frontend.sh start --dev   # `next dev` with hot reload (UI work only)
#   scripts/dev/frontend.sh stop
#   scripts/dev/frontend.sh restart [--dev]
#   scripts/dev/frontend.sh status
#
# Env: FRONTEND_PORT (default 3000); RLM_STUDIO_PORT (backend port, default
# 8000) or INTERNAL_API_URL to point the proxy elsewhere. The proxy target is
# fixed at build time, so restart after changing it.
#
# Runs in the background; log: ${TMPDIR:-/tmp}/rlm-studio-dev/frontend.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"
PORT="${FRONTEND_PORT:-3000}"
export INTERNAL_API_URL="${INTERNAL_API_URL:-http://localhost:${RLM_STUDIO_PORT:-8000}}"
TMP_ROOT="${TMPDIR:-/tmp}"
LOG_DIR="${TMP_ROOT%/}/rlm-studio-dev"
LOG="$LOG_DIR/frontend.log"
MIN_NODE_MAJOR=22
# Only processes whose command line matches this are ever killed.
OWN_PROCESS_PATTERN="next"
START_TIMEOUT_S=120
STOP_TIMEOUT_S=10

listener_pids() {
  lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

is_ours() {
  ps -o command= -p "$1" 2>/dev/null | grep -Eq "$OWN_PROCESS_PATTERN"
}

node_major() {
  node -v 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/' || echo 0
}

ensure_node() {
  if [ "$(node_major)" -lt "$MIN_NODE_MAJOR" ] && [ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    source "${NVM_DIR:-$HOME/.nvm}/nvm.sh" && nvm use "$MIN_NODE_MAJOR" >/dev/null
    set -u
  fi
  if [ "$(node_major)" -lt "$MIN_NODE_MAJOR" ]; then
    echo "frontend: Node >= $MIN_NODE_MAJOR required, found $(node -v 2>/dev/null || echo none)." >&2
    exit 1
  fi
}

status() {
  local pids
  pids="$(listener_pids)"
  if [ -z "$pids" ]; then
    echo "frontend: stopped (nothing listening on :$PORT)"
    return 1
  fi
  echo "frontend: running on :$PORT (pid $(echo $pids)), proxying to $INTERNAL_API_URL"
}

start() {
  local mode="${1:-}" pids
  pids="$(listener_pids)"
  if [ -n "$pids" ]; then
    echo "frontend: :$PORT is already in use (pid $(echo $pids))." >&2
    echo "          Run '$0 stop', or set FRONTEND_PORT to another port." >&2
    exit 1
  fi
  ensure_node
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "frontend: dependencies missing — run 'npm ci --prefix frontend' first." >&2
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  cd "$FRONTEND_DIR"
  if [ "$mode" = "--dev" ]; then
    nohup npm run dev -- -p "$PORT" >"$LOG" 2>&1 &
  else
    echo "frontend: building (proxy target $INTERNAL_API_URL)…"
    npm run build >"$LOG" 2>&1 || { echo "frontend: build failed — see $LOG" >&2; exit 1; }
    nohup npm run start -- -p "$PORT" >>"$LOG" 2>&1 &
  fi
  echo "frontend: starting on :$PORT (log: $LOG)"
  for _ in $(seq "$START_TIMEOUT_S"); do
    if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; then
      echo "frontend: up — http://localhost:$PORT"
      return 0
    fi
    sleep 1
  done
  echo "frontend: did not answer within ${START_TIMEOUT_S}s — see $LOG" >&2
  exit 1
}

stop() {
  local pids pid
  pids="$(listener_pids)"
  if [ -z "$pids" ]; then
    echo "frontend: not running on :$PORT"
    return 0
  fi
  for pid in $pids; do
    if ! is_ours "$pid"; then
      echo "frontend: pid $pid on :$PORT is not a Next.js server — not touching it:" >&2
      ps -o pid=,command= -p "$pid" >&2
      exit 1
    fi
  done
  # shellcheck disable=SC2086
  kill $pids
  for _ in $(seq "$STOP_TIMEOUT_S"); do
    [ -z "$(listener_pids)" ] && { echo "frontend: stopped"; return 0; }
    sleep 1
  done
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  echo "frontend: stopped (forced after ${STOP_TIMEOUT_S}s)"
}

case "${1:-}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  restart) stop; start "${2:-}" ;;
  status) status ;;
  *) echo "usage: $0 {start [--dev]|stop|restart [--dev]|status}" >&2; exit 2 ;;
esac
