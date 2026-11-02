#!/usr/bin/env bash
# backend.sh — run the RLM Studio API server on its own, so it can be stopped
# and restarted while the UI (scripts/dev/frontend.sh) stays up. That is what
# the frontend manual test plan's §1.1, §1.2 and §5.8 need.
#
# Ownership: start records the launched pid, its process group and start time
# in ${TMPDIR}/rlm-studio-dev/backend-<repo-id>.pid. stop signals only that
# group, and only while the recorded start time still matches. It never
# infers ownership from a process name, so another checkout or a manually
# started server on the same port is left alone. A start that fails (bad
# command, health timeout, or a different process answering on the port)
# terminates the process it launched.
#
# Usage:
#   scripts/dev/backend.sh start      # API on :8000 (RLM_STUDIO_PORT to change)
#   scripts/dev/backend.sh stop
#   scripts/dev/backend.sh restart
#   scripts/dev/backend.sh status
#
# Runs in the background; log: ${TMPDIR:-/tmp}/rlm-studio-dev/backend.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=scripts/dev/process.sh
. "$SCRIPT_DIR/process.sh"

NAME="backend"
PORT="${RLM_STUDIO_PORT:-8000}"
LOG_DIR="$(dev_log_dir)"
LOG="$LOG_DIR/backend.log"
PID_FILE="$LOG_DIR/backend-$(repo_id "$REPO_ROOT").pid"
START_TIMEOUT_S=60
STOP_TIMEOUT_S=10

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
  trap cleanup_start EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  require_startable "$NAME" "$PID_FILE" || exit 1
  ensure_port_free "$NAME" "$PORT" \
    "Run '$0 stop', or set RLM_STUDIO_PORT to another port." || exit 1

  mkdir -p "$LOG_DIR"
  cd "$REPO_ROOT"
  set -m
  nohup uv run python -m rlmstudio.server --port "$PORT" >"$LOG" 2>&1 &
  local pid=$! pgid started _ up=0
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
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      up=1
      break
    fi
    sleep 1
  done
  if [ "$up" != "1" ]; then
    echo "$NAME: did not answer /health within ${START_TIMEOUT_S}s — see $LOG" >&2
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
  echo "$NAME: up — http://localhost:$PORT/health"
}

stop() {
  stop_owned "$NAME" "$PID_FILE" "$PORT" "$STOP_TIMEOUT_S"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
