#!/usr/bin/env bash
# process.sh — process-ownership helpers shared by scripts/dev/*.sh.
#
# Ownership is RECORDED, never inferred. `start` writes the launched pid,
# its process-group id and the launcher's start time into a
# repository-specific file under ${TMPDIR}/rlm-studio-dev/. `stop` signals
# that group and nothing else: no command-line pattern decides what may be
# killed, so a server from another checkout — or an unrelated Next.js app on
# the same port — is never touched. A recorded identity whose start time no
# longer matches is treated as stale and is never signalled.

dev_log_dir() {
  local root="${TMPDIR:-/tmp}"
  printf '%s/rlm-studio-dev\n' "${root%/}"
}

repo_id() {
  # One id per checkout root, so two clones cannot share a PID file.
  printf '%s' "$1" | cksum | awk '{print $1}'
}

listener_pids() {
  lsof -nP -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null || true
}

ps_field() {
  ps -o "$2=" -p "$1" 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

listener_in_group() {  # PORT PGID
  local pid
  for pid in $(listener_pids "$1"); do
    if [ "$(ps_field "$pid" pgid)" = "$2" ]; then
      return 0
    fi
  done
  return 1
}

process_identity_matches() {  # PID STARTED
  [ -n "${1:-}" ] && [ -n "${2:-}" ] || return 1
  kill -0 "$1" 2>/dev/null || return 1
  [ "$(ps_field "$1" lstart)" = "$2" ]
}

write_identity() {  # FILE PID PGID STARTED
  local tmp="$1.tmp.$$"
  printf '%s\n%s\n%s\n' "$2" "$3" "$4" >"$tmp"
  mv -f "$tmp" "$1"
}

read_identity() {  # FILE — sets ID_PID / ID_PGID / ID_STARTED
  [ -f "$1" ] || return 1
  ID_PID="$(sed -n '1p' "$1")"
  ID_PGID="$(sed -n '2p' "$1")"
  ID_STARTED="$(sed -n '3p' "$1")"
  case "$ID_PID" in '' | *[!0-9]*) return 1 ;; esac
  case "$ID_PGID" in '' | *[!0-9]*) return 1 ;; esac
  [ -n "$ID_STARTED" ] || return 1
  [ "$ID_PID" -gt 0 ] && [ "$ID_PGID" -gt 0 ]
}

# ownership_state FILE — classify the recorded owner.
#   0 = no record, or the launcher and its group are both gone (a stale file
#       is removed)
#   1 = the recorded launcher is alive with the recorded start time
#   2 = the recorded group is alive but its launcher is gone or changed:
#       the identity cannot be verified, so nothing may be signalled and the
#       record must be preserved
ownership_state() {
  local pid_file="$1" actual_pgid
  if ! read_identity "$pid_file"; then
    return 0
  fi
  if process_identity_matches "$ID_PID" "$ID_STARTED"; then
    # The stored PGID is a separate field and must belong to this launcher:
    # a record pairing a live PID with an unrelated group must never become
    # signalable just because the PID and start time happen to match.
    actual_pgid="$(ps_field "$ID_PID" pgid)"
    if [ "$actual_pgid" = "$ID_PGID" ]; then
      return 1
    fi
    return 2
  fi
  if kill -0 -- -"$ID_PGID" 2>/dev/null; then
    return 2
  fi
  rm -f "$pid_file"
  return 0
}

terminate_group() {  # PGID TIMEOUT_S
  local pgid="$1" timeout="$2" _
  if ! kill -TERM -- -"$pgid" 2>/dev/null; then
    # Either the group is already gone, or signalling it was not permitted.
    # Never report success while it still exists.
    ! kill -0 -- -"$pgid" 2>/dev/null
    return $?
  fi
  for _ in $(seq "$timeout"); do
    kill -0 -- -"$pgid" 2>/dev/null || return 0
    sleep 1
  done
  kill -KILL -- -"$pgid" 2>/dev/null || true
  sleep 1
  ! kill -0 -- -"$pgid" 2>/dev/null
}

stop_owned() {  # NAME PID_FILE PORT TIMEOUT_S
  local name="$1" pid_file="$2" port="$3" timeout="$4" pids state=0
  ownership_state "$pid_file" || state=$?
  case "$state" in
    0)
      echo "$name: not running (no live record for this checkout)"
      pids="$(listener_pids "$port")"
      if [ -n "$pids" ]; then
        echo "$name: note: :$port is held by pid $(echo $pids), which this script did not start; leaving it alone" >&2
      fi
      return 0
      ;;
    1)
      if terminate_group "$ID_PGID" "$timeout"; then
        rm -f "$pid_file"
        echo "$name: stopped (pid $ID_PID, group $ID_PGID)"
        return 0
      fi
      echo "$name: process group $ID_PGID is still alive after SIGKILL." >&2
      echo "        Keeping the ownership record at $pid_file." >&2
      return 1
      ;;
    *)
      echo "$name: refusing to signal anything: the ownership record cannot be verified." >&2
      echo "        Launcher pid $ID_PID / start time or its group $ID_PGID no longer match." >&2
      echo "        Inspect: ps -o pid=,pgid=,command= -p $ID_PID" >&2
      echo "                 ps -o pid=,pgid=,command= -g $ID_PGID" >&2
      return 1
      ;;
  esac
}

status_owned() {  # NAME PID_FILE PORT
  local name="$1" pid_file="$2" port="$3" pids state=0
  ownership_state "$pid_file" || state=$?
  case "$state" in
    0)
      echo "$name: stopped (no live record for this checkout)"
      pids="$(listener_pids "$port")"
      if [ -n "$pids" ]; then
        echo "$name: note: :$port is held by pid $(echo $pids), which this script did not start" >&2
      fi
      return 1
      ;;
    1)
      if listener_in_group "$port" "$ID_PGID"; then
        echo "$name: running on :$port (pid $ID_PID, group $ID_PGID)"
      else
        echo "$name: running (pid $ID_PID, group $ID_PGID) but not listening on :$port"
      fi
      return 0
      ;;
    *)
      echo "$name: unknown — the ownership record cannot be verified" >&2
      echo "        (launcher pid $ID_PID, group $ID_PGID). Inspect:" >&2
      echo "        ps -o pid=,pgid=,command= -p $ID_PID" >&2
      return 1
      ;;
  esac
}

require_startable() {  # NAME PID_FILE
  local name="$1" pid_file="$2" state=0
  ownership_state "$pid_file" || state=$?
  case "$state" in
    0) return 0 ;;
    1)
      echo "$name: already running (pid $ID_PID, group $ID_PGID). Use '$0 restart'." >&2
      return 1
      ;;
    *)
      echo "$name: refusing to start: the ownership record cannot be verified." >&2
      echo "        Launcher pid $ID_PID / start time or its group $ID_PGID no longer match." >&2
      echo "        Inspect: ps -o pid=,pgid=,command= -p $ID_PID" >&2
      echo "                 ps -o pid=,pgid=,command= -g $ID_PGID" >&2
      echo "        The ownership record at $pid_file was kept." >&2
      return 1
      ;;
  esac
}

ensure_port_free() {  # NAME PORT HINT
  local name="$1" port="$2" hint="$3" pids
  pids="$(listener_pids "$port")"
  if [ -n "$pids" ]; then
    echo "$name: :$port is in use by pid $(echo $pids), which this script did not start." >&2
    echo "        Refusing to touch it. $hint" >&2
    # shellcheck disable=SC2086  # word splitting is the point: ps wants each pid
    ps -o pid=,command= -p $pids >&2 2>/dev/null || true
    return 1
  fi
  return 0
}
