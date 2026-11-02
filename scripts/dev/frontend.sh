#!/usr/bin/env bash
# frontend.sh — run the RLM Studio UI as its own process on :3000, proxying
# /api, /health and /ws to the backend (scripts/dev/backend.sh). Stopping one
# leaves the other running.
#
# Ownership: start records the launched pid, its process group and start time
# in ${TMPDIR}/rlm-studio-dev/frontend-<repo-id>.pid. stop signals only that
# group, and only while the recorded start time still matches — an unrelated
# Next.js app on the port is never touched. A start that fails (build error,
# readiness timeout, or a different process answering on the port) terminates
# the process it launched.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/dev/process.sh
. "$SCRIPT_DIR/process.sh"

NAME="frontend"
FRONTEND_DIR="$REPO_ROOT/frontend"
PORT="${FRONTEND_PORT:-3000}"
export INTERNAL_API_URL="${INTERNAL_API_URL:-http://localhost:${RLM_STUDIO_PORT:-8000}}"
LOG_DIR="$(dev_log_dir)"
LOG="$LOG_DIR/frontend.log"
PID_FILE="$LOG_DIR/frontend-$(repo_id "$REPO_ROOT").pid"
MIN_NODE_MAJOR=22
START_TIMEOUT_S=120
STOP_TIMEOUT_S=10

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

cleanup_start() {
  if [ "${START_OK:-0}" = "1" ]; then
    return 0
  fi
  if [ -n "${LAUNCHED_PGID:-}" ]; then
    if terminate_group "$LAUNCHED_PGID" "$STOP_TIMEOUT_S"; then
      rm -f "$PID_FILE"
    else
      echo "$NAME: cleanup could not terminate process group $LAUNCHED_PGID." >&2
      echo "        Keeping the ownership record at $PID_FILE; inspect with:" >&2
      echo "        ps -o pid=,pgid=,command= -g $LAUNCHED_PGID" >&2
    fi
  fi
}

status() {
  status_owned "$NAME" "$PID_FILE" "$PORT"
}

start() {
  local mode="${1:-}" pid pgid started _ up=0
  trap cleanup_start EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  require_startable "$NAME" "$PID_FILE" || exit 1
  ensure_port_free "$NAME" "$PORT" \
    "Run '$0 stop', or set FRONTEND_PORT to another port." || exit 1
  ensure_node
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo "$NAME: dependencies missing — run 'npm ci --prefix frontend' first." >&2
    exit 1
  fi

  mkdir -p "$LOG_DIR"
  cd "$FRONTEND_DIR"
  if [ "$mode" = "--dev" ]; then
    set -m
    nohup npm run dev -- -p "$PORT" >"$LOG" 2>&1 &
  else
    echo "$NAME: building (proxy target $INTERNAL_API_URL)…"
    npm run build >"$LOG" 2>&1 || { echo "$NAME: build failed — see $LOG" >&2; exit 1; }
    set -m
    nohup npm run start -- -p "$PORT" >>"$LOG" 2>&1 &
  fi
  pid=$!
  pgid="$(ps_field "$pid" pgid)"
  started="$(ps_field "$pid" lstart)"
  if [ -z "$pgid" ] || [ -z "$started" ]; then
    echo "$NAME: launched process exited immediately — see $LOG" >&2
    exit 1
  fi
  LAUNCHED_PGID="$pgid"
  write_identity "$PID_FILE" "$pid" "$pgid" "$started"
  disown || true
  echo "$NAME: starting on :$PORT (pid $pid, log: $LOG)"

  for _ in $(seq "$START_TIMEOUT_S"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$NAME: exited during startup — see $LOG" >&2
      exit 1
    fi
    if curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; then
      up=1
      break
    fi
    sleep 1
  done
  if [ "$up" != "1" ]; then
    echo "$NAME: did not answer within ${START_TIMEOUT_S}s — see $LOG" >&2
    exit 1
  fi
  if ! listener_in_group "$PORT" "$pgid"; then
    echo "$NAME: :$PORT answered, but not from the process this script started." >&2
    echo "        Stopping it; find the other listener with:" >&2
    echo "        lsof -nP -iTCP:$PORT -sTCP:LISTEN" >&2
    exit 1
  fi

  START_OK=1
  trap - EXIT INT TERM
  echo "$NAME: up — http://localhost:$PORT"
}

stop() {
  stop_owned "$NAME" "$PID_FILE" "$PORT" "$STOP_TIMEOUT_S"
}

case "${1:-}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  restart) stop; start "${2:-}" ;;
  status) status ;;
  *) echo "usage: $0 {start [--dev]|stop|restart [--dev]|status}" >&2; exit 2 ;;
esac
