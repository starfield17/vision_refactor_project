#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/start_stats.sh [--config <path>] [--workdir <path>]

Examples:
  scripts/start_stats.sh
  scripts/start_stats.sh --config ./work-dir/config.toml
  scripts/start_stats.sh --workdir ./work-dir
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/work-dir/config.toml"
WORKDIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Config not found: $CONFIG_PATH" >&2
  exit 2
fi

readarray -t CFG_LINES < <(
  python - "$CONFIG_PATH" "$WORKDIR_OVERRIDE" <<'PY'
import sys
from pathlib import Path
from share.config.config_loader import load_config

config_path = Path(sys.argv[1]).resolve()
workdir_override = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
cfg = load_config(config_path=config_path, workdir_override=workdir_override, overrides=[])
stats_cfg = cfg["deploy"]["statistics"]
print(Path(cfg["workspace"]["root"]).resolve())
print(int(stats_cfg["api_port"]))
print(int(stats_cfg["ui_port"]))
PY
)

WORKDIR_ROOT="${CFG_LINES[0]}"
API_PORT="${CFG_LINES[1]}"
UI_PORT="${CFG_LINES[2]}"
TMP_DIR="${WORKDIR_ROOT}/tmp"
mkdir -p "$TMP_DIR"

API_PID_FILE="${TMP_DIR}/statistics_api.pid"
UI_PID_FILE="${TMP_DIR}/statistics_ui.pid"
API_LOG="${TMP_DIR}/statistics_api.log"
UI_LOG="${TMP_DIR}/statistics_ui.log"

COMMON_ARGS=(--config "$CONFIG_PATH")
if [[ -n "$WORKDIR_OVERRIDE" ]]; then
  COMMON_ARGS+=(--workdir "$WORKDIR_OVERRIDE")
fi

is_running_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

start_service_if_needed() {
  local pid_file="$1"
  local log_file="$2"
  shift 2

  local existing_pid=""
  if [[ -f "$pid_file" ]]; then
    existing_pid="$(cat "$pid_file" 2>/dev/null || true)"
  fi
  if is_running_pid "$existing_pid"; then
    echo "$existing_pid"
    return 0
  fi

  nohup "$@" >"$log_file" 2>&1 &
  local new_pid="$!"
  echo "$new_pid" >"$pid_file"
  echo "$new_pid"
}

wait_http_ok() {
  local url="$1"
  local timeout_sec="$2"
  local i=0
  while (( i < timeout_sec )); do
    if python - "$url" <<'PY'
import sys
from urllib.request import urlopen

url = sys.argv[1]
try:
    with urlopen(url, timeout=1.5) as resp:
        ok = int(resp.status) == 200
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi
    sleep 1
    ((i=i+1))
  done
  return 1
}

wait_tcp_open() {
  local host="$1"
  local port="$2"
  local timeout_sec="$3"
  local i=0
  while (( i < timeout_sec )); do
    if python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.2)
try:
    sock.connect((host, port))
    ok = True
except Exception:
    ok = False
finally:
    sock.close()
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi
    sleep 1
    ((i=i+1))
  done
  return 1
}

echo "[INFO] Starting statistics API/UI ..."
API_PID="$(start_service_if_needed "$API_PID_FILE" "$API_LOG" python -m deploy.statistics.api "${COMMON_ARGS[@]}")"
UI_PID="$(start_service_if_needed "$UI_PID_FILE" "$UI_LOG" python -m deploy.statistics.ui "${COMMON_ARGS[@]}")"

if ! wait_http_ok "http://127.0.0.1:${API_PORT}/health" 25; then
  echo "[ERROR] Statistics API health check failed. See log: $API_LOG" >&2
  exit 1
fi
if ! wait_tcp_open "127.0.0.1" "$UI_PORT" 25; then
  echo "[ERROR] Statistics UI port check failed. See log: $UI_LOG" >&2
  exit 1
fi

echo "[OK] Statistics API ready: http://127.0.0.1:${API_PORT}/health"
echo "[OK] Statistics UI ready : http://127.0.0.1:${UI_PORT}"
echo "[INFO] API pid=${API_PID}, log=${API_LOG}"
echo "[INFO] UI  pid=${UI_PID}, log=${UI_LOG}"
echo "[INFO] Stop with:"
echo "  kill \$(cat \"$API_PID_FILE\") \$(cat \"$UI_PID_FILE\")"
