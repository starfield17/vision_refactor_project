#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/stop_stats.sh [--config <path>] [--workdir <path>]

Examples:
  scripts/stop_stats.sh
  scripts/stop_stats.sh --config ./work-dir/config.toml
  scripts/stop_stats.sh --workdir ./work-dir
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

WORKDIR_ROOT="$(python - "$CONFIG_PATH" "$WORKDIR_OVERRIDE" <<'PY'
import sys
from pathlib import Path
from share.config.config_loader import load_config

config_path = Path(sys.argv[1]).resolve()
workdir_override = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
cfg = load_config(config_path=config_path, workdir_override=workdir_override, overrides=[])
print(Path(cfg["workspace"]["root"]).resolve())
PY
)"

TMP_DIR="${WORKDIR_ROOT}/tmp"
API_PID_FILE="${TMP_DIR}/statistics_api.pid"
UI_PID_FILE="${TMP_DIR}/statistics_ui.pid"

is_running_pid() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  if [[ ! -f "$pid_file" ]]; then
    echo "[INFO] ${label}: pid file not found (${pid_file})"
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if ! is_running_pid "$pid"; then
    echo "[INFO] ${label}: not running (stale pid file removed)"
    rm -f "$pid_file"
    return 0
  fi

  echo "[INFO] ${label}: stopping pid=${pid}"
  kill "$pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    if ! is_running_pid "$pid"; then
      break
    fi
    sleep 0.5
  done
  if is_running_pid "$pid"; then
    echo "[WARN] ${label}: force killing pid=${pid}"
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$pid_file"
}

stop_pid_file "statistics-api" "$API_PID_FILE"
stop_pid_file "statistics-ui" "$UI_PID_FILE"
echo "[OK] stop_stats done"
