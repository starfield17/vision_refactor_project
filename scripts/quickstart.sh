#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
WEB_PORT="${CONTROL_PLANE_WEB_PORT:-5173}"
LOG_TAIL="${QUICKSTART_LOG_TAIL:-120}"
RUN_DIR="${PROJECT_DIR}/work-dir/tmp/quickstart"
COMMAND="${1:-up}"

usage() {
  cat <<'EOF'
Usage: bash scripts/quickstart.sh [command] [args]

Local commands:
  up              Start local Control Plane, core workers, statistics, and Web UI
  down            Stop local quickstart services
  restart         Stop then start local quickstart services
  status          Show process and health status
  logs [service]  Tail a service log, or list services when omitted

Podman commands:
  podman <command> [profile]
                  Proxy to deployments/install.sh. Example:
                  bash scripts/quickstart.sh podman up all-in-one

Environment:
  PYTHON=/path/to/python
  CONTROL_PLANE_WEB_PORT=5173
  QUICKSTART_REMOTE=1
  QUICKSTART_LOG_TAIL=120
EOF
}

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

die() {
  echo "[ERROR] $*" >&2
  exit 2
}

prepare_dirs() {
  mkdir -p "${PROJECT_DIR}/work-dir"/{artifacts,state,stats,tmp,models,datasets,runs,outputs}
  mkdir -p "$RUN_DIR"
}

service_names() {
  printf '%s\n' \
    control-plane \
    statistics \
    train-worker \
    autolabel-worker \
    edge-agent \
    control-plane-web
  if [[ "${QUICKSTART_REMOTE:-0}" == "1" ]]; then
    printf '%s\n' remote-worker
  fi
}

pid_file() {
  printf '%s/%s.pid\n' "$RUN_DIR" "$1"
}

log_file() {
  printf '%s/%s.log\n' "$RUN_DIR" "$1"
}

health_url() {
  case "$1" in
    control-plane) echo "http://127.0.0.1:7800/health" ;;
    statistics) echo "http://127.0.0.1:7803/health" ;;
    train-worker) echo "http://127.0.0.1:7811/health" ;;
    autolabel-worker) echo "http://127.0.0.1:7812/health" ;;
    edge-agent) echo "http://127.0.0.1:7813/health" ;;
    remote-worker) echo "http://127.0.0.1:60051/health" ;;
    control-plane-web) echo "http://127.0.0.1:${WEB_PORT}" ;;
    *) return 1 ;;
  esac
}

service_port() {
  case "$1" in
    control-plane) echo "7800" ;;
    statistics) echo "7803" ;;
    train-worker) echo "7811" ;;
    autolabel-worker) echo "7812" ;;
    edge-agent) echo "7813" ;;
    remote-worker) echo "60051" ;;
    control-plane-web) echo "$WEB_PORT" ;;
    *) return 1 ;;
  esac
}

service_host() {
  case "$1" in
    control-plane-web) echo "0.0.0.0" ;;
    *) echo "127.0.0.1" ;;
  esac
}

is_running() {
  local service="$1"
  local file
  file="$(pid_file "$service")"
  [[ -f "$file" ]] || return 1
  local pid
  pid="$(cat "$file")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1 || kill -0 -- "-$pid" >/dev/null 2>&1
}

assert_port_available() {
  local service="$1"
  local port
  port="$(service_port "$service")"
  local host
  host="$(service_host "$service")"
  "$PYTHON_BIN" - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError as exc:
    print(f"{host}:{port} is not available: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
PY
}

run_health_check() {
  local url="$1"
  "$PYTHON_BIN" - "$url" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import ProxyHandler, build_opener

opener = build_opener(ProxyHandler({}))
with opener.open(sys.argv[1], timeout=1.5) as response:
    if response.status >= 500:
        raise SystemExit(1)
PY
}

wait_for_health() {
  local service="$1"
  local url
  url="$(health_url "$service")"
  local attempt
  for attempt in {1..30}; do
    if run_health_check "$url"; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

service_command() {
  local service="$1"
  case "$service" in
    control-plane)
      printf '%q ' "$PYTHON_BIN" -m control_plane.api \
        --config control_plane/config/config.example.toml
      ;;
    statistics)
      printf '%q ' "$PYTHON_BIN" -m stats_service.api \
        --config stats_service/config/config.example.toml \
        --set control_plane.url=http://127.0.0.1:7800 \
        --set server.advertise_url=http://127.0.0.1:7803
      ;;
    train-worker)
      printf '%q ' "$PYTHON_BIN" -m train_worker.service \
        --config train_worker/config/config.example.toml \
        --set control_plane.url=http://127.0.0.1:7800 \
        --set server.advertise_url=http://127.0.0.1:7811
      ;;
    autolabel-worker)
      printf '%q ' "$PYTHON_BIN" -m autolabel_worker.service \
        --config autolabel_worker/config/config.example.toml \
        --set control_plane.url=http://127.0.0.1:7800 \
        --set server.advertise_url=http://127.0.0.1:7812
      ;;
    edge-agent)
      printf '%q ' "$PYTHON_BIN" -m edge_agent.service \
        --config edge_agent/config/config.example.toml \
        --set control_plane.url=http://127.0.0.1:7800 \
        --set server.advertise_url=http://127.0.0.1:7813
      ;;
    remote-worker)
      printf '%q ' "$PYTHON_BIN" -m remote_worker.api \
        --config remote_worker/config/config.example.toml \
        --set control_plane.url=http://127.0.0.1:7800 \
        --set node.endpoint=http://127.0.0.1:60051
      ;;
    control-plane-web)
      printf '%q ' env VITE_CONTROL_PLANE_API_URL="http://127.0.0.1:7800" \
        npm --prefix control_plane/web run dev -- --port "$WEB_PORT" --strictPort
      ;;
    *)
      die "unknown service: $service"
      ;;
  esac
}

start_service() {
  local service="$1"
  if is_running "$service"; then
    info "${service} already running (pid $(cat "$(pid_file "$service")"))"
    return 0
  fi
  if ! assert_port_available "$service"; then
    die "${service} port $(service_port "$service") on $(service_host "$service") is already in use"
  fi
  local cmd log pid
  cmd="$(service_command "$service")"
  log="$(log_file "$service")"
  info "Starting ${service}"
  local quickstart_no_proxy="127.0.0.1,localhost,::1"
  if [[ -n "${NO_PROXY:-}" ]]; then
    quickstart_no_proxy="${quickstart_no_proxy},${NO_PROXY}"
  fi
  if [[ -n "${no_proxy:-}" ]]; then
    quickstart_no_proxy="${quickstart_no_proxy},${no_proxy}"
  fi
  pushd "$PROJECT_DIR" >/dev/null
  if command -v setsid >/dev/null 2>&1; then
    NO_PROXY="$quickstart_no_proxy" no_proxy="$quickstart_no_proxy" \
      setsid nohup bash -lc "exec ${cmd}" >"$log" 2>&1 < /dev/null &
  else
    NO_PROXY="$quickstart_no_proxy" no_proxy="$quickstart_no_proxy" \
      nohup bash -lc "exec ${cmd}" >"$log" 2>&1 < /dev/null &
  fi
  pid="$!"
  popd >/dev/null
  echo "$pid" >"$(pid_file "$service")"
  sleep 0.2
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    warn "${service} exited during startup; log: $log"
    tail -n 40 "$log" >&2 || true
    return 1
  fi
  if wait_for_health "$service"; then
    info "${service} ready at $(health_url "$service")"
  else
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$(pid_file "$service")"
      warn "${service} exited before health check passed; log: $log"
      tail -n 40 "$log" >&2 || true
      return 1
    fi
    warn "${service} started but health check did not pass yet; log: $log"
  fi
}

stop_service() {
  local service="$1"
  local file pid
  file="$(pid_file "$service")"
  if [[ ! -f "$file" ]]; then
    info "${service} not started by quickstart"
    return 0
  fi
  pid="$(cat "$file")"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" >/dev/null 2>&1; then
    if ! kill -0 -- "-$pid" >/dev/null 2>&1; then
      rm -f "$file"
      info "${service} not running"
      return 0
    fi
  fi
  info "Stopping ${service} (pid ${pid})"
  kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
  local attempt
  for attempt in {1..20}; do
    if ! kill -0 "$pid" >/dev/null 2>&1 && ! kill -0 -- "-$pid" >/dev/null 2>&1; then
      rm -f "$file"
      return 0
    fi
    sleep 0.2
  done
  warn "${service} did not stop after TERM; sending KILL"
  kill -9 -- "-$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$file"
}

up() {
  prepare_dirs
  local service
  for service in $(service_names); do
    start_service "$service"
  done
  cat <<EOF
[OK] Quickstart services are running.
Control Plane API: http://127.0.0.1:7800
Control Plane Web: http://127.0.0.1:${WEB_PORT}
Logs: bash scripts/quickstart.sh logs <service>
Stop: bash scripts/quickstart.sh down
EOF
}

down() {
  prepare_dirs
  local services=()
  local service
  while read -r service; do
    services+=("$service")
  done < <(service_names)
  local index
  for ((index=${#services[@]} - 1; index >= 0; index--)); do
    stop_service "${services[$index]}"
  done
}

status() {
  prepare_dirs
  printf '%-20s %-8s %-8s %s\n' "SERVICE" "PID" "HEALTH" "URL"
  local service pid state url
  for service in $(service_names); do
    pid="-"
    if is_running "$service"; then
      pid="$(cat "$(pid_file "$service")")"
    fi
    url="$(health_url "$service")"
    if run_health_check "$url"; then
      state="ok"
    else
      state="down"
    fi
    printf '%-20s %-8s %-8s %s\n' "$service" "$pid" "$state" "$url"
  done
}

logs() {
  prepare_dirs
  local service="${1:-}"
  if [[ -z "$service" ]]; then
    service_names
    return 0
  fi
  local log
  log="$(log_file "$service")"
  [[ -f "$log" ]] || die "no log file for ${service}: ${log}"
  tail -n "$LOG_TAIL" -f "$log"
}

podman_proxy() {
  local subcommand="${1:-up}"
  local profile="${2:-all-in-one}"
  exec bash "${PROJECT_DIR}/deployments/install.sh" "$subcommand" "$profile"
}

case "$COMMAND" in
  -h|--help|help)
    usage
    ;;
  up)
    up
    ;;
  down)
    down
    ;;
  restart)
    down
    up
    ;;
  status)
    status
    ;;
  logs)
    shift || true
    logs "${1:-}"
    ;;
  podman)
    shift || true
    podman_proxy "${1:-up}" "${2:-all-in-one}"
    ;;
  *)
    usage >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
