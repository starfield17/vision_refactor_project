#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/status_stats.sh [--config <path>] [--workdir <path>]

Examples:
  scripts/status_stats.sh
  scripts/status_stats.sh --config ./work-dir/config.toml
  scripts/status_stats.sh --workdir ./work-dir
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
API_PID_FILE="${TMP_DIR}/statistics_api.pid"
UI_PID_FILE="${TMP_DIR}/statistics_ui.pid"

is_running_pid() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

check_port() {
  local host="$1"
  local port="$2"
  python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.2)
    try:
        sock.connect((host, port))
        ok = True
    except Exception:
        ok = False
    finally:
        sock.close()
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

check_api_health() {
  local port="$1"
  python - "$port" <<'PY'
import json
import sys
from urllib.request import urlopen

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/health"
try:
    with urlopen(url, timeout=1.8) as resp:
        if int(resp.status) != 200:
            raise SystemExit(1)
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
        if not bool(body.get("ok")):
            raise SystemExit(1)
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

api_pid=""
ui_pid=""
[[ -f "$API_PID_FILE" ]] && api_pid="$(cat "$API_PID_FILE" 2>/dev/null || true)"
[[ -f "$UI_PID_FILE" ]] && ui_pid="$(cat "$UI_PID_FILE" 2>/dev/null || true)"

api_running=false
ui_running=false
if is_running_pid "$api_pid"; then
  api_running=true
fi
if is_running_pid "$ui_pid"; then
  ui_running=true
fi

api_health=false
ui_port_open=false
if check_api_health "$API_PORT"; then
  api_health=true
fi
if check_port "127.0.0.1" "$UI_PORT"; then
  ui_port_open=true
fi

echo "config=${CONFIG_PATH}"
echo "workdir=${WORKDIR_ROOT}"
echo "api_port=${API_PORT} ui_port=${UI_PORT}"
echo "statistics-api: pid=${api_pid:-none} running=${api_running} health=${api_health}"
echo "statistics-ui : pid=${ui_pid:-none} running=${ui_running} port_open=${ui_port_open}"

if $api_running && $ui_running && $api_health && $ui_port_open; then
  echo "[OK] statistics services look healthy"
  exit 0
fi

echo "[WARN] statistics services are partially unavailable"
exit 1
