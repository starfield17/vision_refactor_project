#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/start_train_autolabel.sh [--config <path>] [--workdir <path>] [--ui-port <port>] [--env-file <path>] [--python-bin <path>]

Examples:
  scripts/start_train_autolabel.sh
  scripts/start_train_autolabel.sh --config ./work-dir/config.toml
  scripts/start_train_autolabel.sh --workdir ./work-dir --ui-port 7794
  PYTHON_BIN=/home/hazel/miniconda3/envs/Lab/bin/python scripts/start_train_autolabel.sh
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/work-dir/config.toml"
WORKDIR_OVERRIDE=""
UI_PORT="7794"
ENV_FILE="${ROOT_DIR}/work-dir/secrets/llm.env"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
    --ui-port)
      UI_PORT="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
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

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  echo "[INFO] Loaded environment file: $ENV_FILE"
elif [[ "$ENV_FILE" != "${ROOT_DIR}/work-dir/secrets/llm.env" ]]; then
  echo "[ERROR] Environment file not found: $ENV_FILE" >&2
  exit 2
fi

readarray -t CFG_LINES < <(
  "$PYTHON_BIN" - "$CONFIG_PATH" "$WORKDIR_OVERRIDE" <<'PY'
import sys
from pathlib import Path

from share.application.api_common import resolve_api_token
from share.config.config_loader import load_config

config_path = Path(sys.argv[1]).resolve()
workdir_override = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
cfg = load_config(config_path=config_path, workdir_override=workdir_override, overrides=[])
svc_cfg = cfg["services"]["train_autolabel"]
host = str(svc_cfg["host"])
api_url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
print(Path(cfg["workspace"]["root"]).resolve())
print(host)
print(api_url_host)
print(int(svc_cfg["port"]))
print(resolve_api_token(svc_cfg))
PY
)

WORKDIR_ROOT="${CFG_LINES[0]}"
API_HOST="${CFG_LINES[1]}"
API_URL_HOST="${CFG_LINES[2]}"
API_PORT="${CFG_LINES[3]}"
API_TOKEN="${CFG_LINES[4]}"
TMP_DIR="${WORKDIR_ROOT}/tmp"
mkdir -p "$TMP_DIR"

API_PID_FILE="${TMP_DIR}/train_autolabel_api.pid"
UI_PID_FILE="${TMP_DIR}/train_autolabel_ui.pid"
API_LOG="${TMP_DIR}/train_autolabel_api.log"
UI_LOG="${TMP_DIR}/train_autolabel_ui.log"
API_BASE_URL="http://${API_URL_HOST}:${API_PORT}"

COMMON_ARGS=(--config "$CONFIG_PATH")
if [[ -n "$WORKDIR_OVERRIDE" ]]; then
  COMMON_ARGS+=(--workdir "$WORKDIR_OVERRIDE")
fi

is_running_pid() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

wait_http_ok() {
  local url="$1"
  local timeout_sec="$2"
  local i=0
  while (( i < timeout_sec )); do
    if "$PYTHON_BIN" - "$url" "$API_TOKEN" <<'PY'
import sys
from urllib.request import Request, urlopen

url = sys.argv[1]
token = sys.argv[2]
headers = {}
if token:
    headers["Authorization"] = f"Bearer {token}"
try:
    with urlopen(Request(url, headers=headers), timeout=1.5) as resp:
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
    if "$PYTHON_BIN" - "$host" "$port" <<'PY'
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

start_api_if_needed() {
  local existing_pid=""
  if [[ -f "$API_PID_FILE" ]]; then
    existing_pid="$(cat "$API_PID_FILE" 2>/dev/null || true)"
  fi
  if is_running_pid "$existing_pid"; then
    echo "$existing_pid"
    return 0
  fi
  if wait_http_ok "${API_BASE_URL}/health" 1; then
    rm -f "$API_PID_FILE"
    echo "external"
    return 0
  fi

  nohup "$PYTHON_BIN" -m services.train_autolabel.api "${COMMON_ARGS[@]}" >"$API_LOG" 2>&1 &
  local new_pid="$!"
  echo "$new_pid" >"$API_PID_FILE"
  echo "$new_pid"
}

start_ui_if_needed() {
  local existing_pid=""
  if [[ -f "$UI_PID_FILE" ]]; then
    existing_pid="$(cat "$UI_PID_FILE" 2>/dev/null || true)"
  fi
  if is_running_pid "$existing_pid"; then
    echo "$existing_pid"
    return 0
  fi
  if wait_tcp_open "127.0.0.1" "$UI_PORT" 1; then
    rm -f "$UI_PID_FILE"
    echo "external"
    return 0
  fi

  (
    cd "${ROOT_DIR}/web/train_autolabel"
    VITE_TRAIN_AUTOLABEL_API_URL="$API_BASE_URL" \
      VITE_TRAIN_AUTOLABEL_API_TOKEN="$API_TOKEN" \
      setsid ./node_modules/.bin/vite --host 0.0.0.0 --port "$UI_PORT" \
      </dev/null >"$UI_LOG" 2>&1 &
    echo "$!" >"$UI_PID_FILE"
  )
  local new_pid
  new_pid="$(cat "$UI_PID_FILE" 2>/dev/null || true)"
  echo "$new_pid"
}

echo "[INFO] Starting train/autolabel backend + React UI ..."
API_PID="$(start_api_if_needed)"
UI_PID="$(start_ui_if_needed)"

if ! wait_http_ok "${API_BASE_URL}/health" 25; then
  echo "[ERROR] Train/autolabel API health check failed. See log: $API_LOG" >&2
  exit 1
fi
if ! wait_tcp_open "127.0.0.1" "$UI_PORT" 25; then
  echo "[ERROR] Train/autolabel UI port check failed. See log: $UI_LOG" >&2
  exit 1
fi

echo "[OK] Train/autolabel API ready: ${API_BASE_URL}/health"
echo "[OK] Train/autolabel UI ready : http://127.0.0.1:${UI_PORT}"
echo "[INFO] API pid=${API_PID}, log=${API_LOG}"
echo "[INFO] UI  pid=${UI_PID}, log=${UI_LOG}"
if [[ "$API_PID" == "external" || "$UI_PID" == "external" ]]; then
  echo "[INFO] external means the port was already serving before this script started."
fi
echo "[INFO] Stop managed processes with:"
echo "  scripts/stop_train_autolabel.sh --config \"$CONFIG_PATH\""
