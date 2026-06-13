#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
WEB_PORT="${CONTROL_PLANE_WEB_PORT:-5173}"
RUN_DIR="${PROJECT_DIR}/work-dir/tmp/quickstart"

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

service_names() {
  printf '%s\n' \
    control-plane \
    statistics \
    edge-agent \
    control-plane-web \
    remote-worker
}

service_port() {
  case "$1" in
    control-plane) echo "7800" ;;
    statistics) echo "7803" ;;
    edge-agent) echo "7813" ;;
    remote-worker) echo "60051" ;;
    control-plane-web) echo "$WEB_PORT" ;;
    *) return 1 ;;
  esac
}

pid_file() {
  printf '%s/%s.pid\n' "$RUN_DIR" "$1"
}

is_project_process() {
  local pid="$1"
  local command
  command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  [[ -n "$command" ]] || return 1
  case "$command" in
    *" -m control_plane.api"*|\
    *" -m stats_service.api"*|\
    *" -m edge_agent.service"*|\
    *" -m remote_worker.api"*|\
    *"npm --prefix control_plane/web run dev"*|\
    *"npm run dev -- --port ${WEB_PORT} --strictPort"*|\
    *"vite --host 0.0.0.0 --port ${WEB_PORT}"*|\
    *"node_modules/.bin/vite"*" --port ${WEB_PORT}"*|\
    *"node_modules/vite/bin/vite.js"*" --port ${WEB_PORT}"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

terminate_pid() {
  local pid="$1"
  local label="$2"
  [[ -n "$pid" ]] || return 0
  if ! kill -0 "$pid" >/dev/null 2>&1 && ! kill -0 -- "-$pid" >/dev/null 2>&1; then
    return 0
  fi
  info "Terminating ${label} (pid ${pid})"
  kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
  local attempt
  for attempt in {1..20}; do
    if ! kill -0 "$pid" >/dev/null 2>&1 && ! kill -0 -- "-$pid" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  warn "${label} did not stop after TERM; sending KILL"
  kill -9 -- "-$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
}

listener_pids() {
  local port="$1"
  "$PYTHON_BIN" - "$port" <<'PY'
import re
import subprocess
import sys

port = sys.argv[1]
try:
    output = subprocess.check_output(
        ["ss", "-ltnp", f"sport = :{port}"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    raise SystemExit(0)

for pid in sorted(set(re.findall(r"pid=(\d+)", output))):
    print(pid)
PY
}

terminate_pid_files() {
  [[ -d "$RUN_DIR" ]] || return 0
  local service file pid
  for service in $(service_names); do
    file="$(pid_file "$service")"
    [[ -f "$file" ]] || continue
    pid="$(cat "$file" 2>/dev/null || true)"
    terminate_pid "$pid" "$service"
    rm -f "$file"
  done
}

terminate_known_ports() {
  local service port pid
  for service in $(service_names); do
    port="$(service_port "$service")"
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      if is_project_process "$pid"; then
        terminate_pid "$pid" "${service} listener on port ${port}"
      else
        warn "Skipping pid ${pid} on port ${port}; command does not match quickstart services"
      fi
    done < <(listener_pids "$port")
  done
}

terminate_known_processes() {
  local pattern pid
  local patterns=(
    "python -m control_plane.api"
    "python -m stats_service.api"
    "python -m edge_agent.service"
    "python -m remote_worker.api"
    "npm --prefix control_plane/web run dev"
    "npm run dev -- --port ${WEB_PORT} --strictPort"
    "vite --host 0.0.0.0 --port ${WEB_PORT}"
    "node_modules/.bin/vite"
    "node_modules/vite/bin/vite.js"
  )
  for pattern in "${patterns[@]}"; do
    while read -r pid; do
      [[ -n "$pid" ]] || continue
      is_project_process "$pid" && terminate_pid "$pid" "$pattern"
    done < <(pgrep -f "$pattern" || true)
  done
}

main() {
  mkdir -p "$RUN_DIR"
  terminate_pid_files
  terminate_known_ports
  terminate_known_processes
  info "Terminate complete"
}

main "$@"
