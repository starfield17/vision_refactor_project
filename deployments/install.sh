#!/usr/bin/env bash
set -euo pipefail

DEPLOYMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${DEPLOYMENTS_DIR}/.." && pwd)"
PROFILES=(all-in-one edge remote-gpu)
COMMAND="${1:-up}"
PROFILE="${2:-all-in-one}"

if [[ "$COMMAND" == "all-in-one" || "$COMMAND" == "edge" || "$COMMAND" == "remote-gpu" ]]; then
  PROFILE="$COMMAND"
  COMMAND="up"
fi

COMPOSE=()
COMPOSE_FILE=""
ENV_FILE=""

usage() {
  cat <<'EOF'
Usage: bash deployments/install.sh [command] [profile]

Commands:
  install   Prepare the profile environment and start containers
  up        Prepare the profile environment and start containers
  down      Stop and remove profile containers
  restart   Restart profile containers
  status    Show profile container status
  logs      Follow profile logs
  doctor    Validate local Podman/profile prerequisites
  config    Render the resolved compose config

Profiles:
  all-in-one
  edge
  remote-gpu

Defaults: command=up, profile=all-in-one
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 2
}

info() {
  echo "[INFO] $*"
}

ok() {
  echo "[OK] $*"
}

contains_profile() {
  local candidate="$1"
  local profile
  for profile in "${PROFILES[@]}"; do
    [[ "$profile" == "$candidate" ]] && return 0
  done
  return 1
}

resolve_compose() {
  if podman compose version >/dev/null 2>&1; then
    COMPOSE=(podman compose)
  elif command -v podman-compose >/dev/null 2>&1; then
    COMPOSE=(podman-compose)
  else
    die "podman compose or podman-compose is required"
  fi
}

random_token() {
  "${PYTHON:-python}" - <<'PY'
import secrets

print(secrets.token_urlsafe(32))
PY
}

ensure_env_key() {
  local key="$1"
  local value="${2:-}"
  if grep -q "^${key}=" "$ENV_FILE"; then
    if [[ -n "$value" ]]; then
      "${PYTHON:-python}" - "$ENV_FILE" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text(encoding="utf-8").splitlines()
updated = [f"{key}={value}" if line.startswith(f"{key}=") and line == f"{key}=" else line for line in lines]
path.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY
    fi
  else
    printf '%s=%s\n' "$key" "$value" >>"$ENV_FILE"
  fi
}

ensure_env() {
  local example="${DEPLOYMENTS_DIR}/${PROFILE}/.env.example"
  ENV_FILE="${DEPLOYMENTS_DIR}/${PROFILE}/.env"
  if [[ ! -f "$ENV_FILE" ]]; then
    [[ -f "$example" ]] || die "missing env example: $example"
    cp "$example" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    info "Created ${ENV_FILE}"
  fi

  case "$PROFILE" in
    all-in-one)
      ensure_env_key CONTROL_PLANE_API_TOKEN "$(random_token)"
      ensure_env_key STATISTICS_API_TOKEN "$(random_token)"
      ensure_env_key EDGE_AGENT_API_TOKEN "$(random_token)"
      ;;
    edge)
      ensure_env_key CONTROL_PLANE_API_TOKEN
      ensure_env_key EDGE_AGENT_API_TOKEN "$(random_token)"
      ;;
    remote-gpu)
      ensure_env_key CONTROL_PLANE_API_TOKEN
      ensure_env_key REMOTE_WORKER_INGEST_API_TOKEN "$(random_token)"
      ;;
  esac
}

prepare_workdir() {
  mkdir -p "${PROJECT_DIR}/work-dir"/{artifacts,state,stats,tmp,models,datasets,runs,outputs}
}

load_env() {
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

port_value() {
  local key="$1"
  case "$key" in
    CONTROL_PLANE_PORT) echo "${CONTROL_PLANE_PORT:-7800}" ;;
    CONTROL_PLANE_WEB_PORT) echo "${CONTROL_PLANE_WEB_PORT:-7801}" ;;
    STATISTICS_PORT) echo "${STATISTICS_PORT:-7803}" ;;
    EDGE_AGENT_PORT) echo "${EDGE_AGENT_PORT:-7813}" ;;
    REMOTE_WORKER_PORT) echo "${REMOTE_WORKER_PORT:-60051}" ;;
    *) return 1 ;;
  esac
}

profile_ports() {
  case "$PROFILE" in
    all-in-one)
      printf '%s\n' CONTROL_PLANE_PORT CONTROL_PLANE_WEB_PORT STATISTICS_PORT EDGE_AGENT_PORT
      ;;
    edge)
      printf '%s\n' EDGE_AGENT_PORT
      ;;
    remote-gpu)
      printf '%s\n' REMOTE_WORKER_PORT
      ;;
  esac
}

check_port() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    if ss -ltn "( sport = :${port} )" | tail -n +2 | grep -q .; then
      echo "[WARN] Port ${port} already appears to be in use"
    fi
  elif command -v lsof >/dev/null 2>&1; then
    if lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "[WARN] Port ${port} already appears to be in use"
    fi
  else
    echo "[WARN] Cannot check port ${port}: ss/lsof not found"
  fi
}

doctor() {
  command -v podman >/dev/null 2>&1 || die "podman is required"
  resolve_compose
  ensure_env
  prepare_workdir
  load_env

  local key port
  while read -r key; do
    port="$(port_value "$key")"
    check_port "$port"
  done < <(profile_ports)

  if [[ "$PROFILE" == "remote-gpu" ]]; then
    if ! podman info --format '{{json .Host}}' | grep -qi 'nvidia\|gpu\|cdi'; then
      echo "[WARN] Could not confirm GPU/CDI support from podman info"
    fi
  fi

  "${COMPOSE[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" config >/dev/null
  ok "${PROFILE} deployment profile is valid"
}

run_compose() {
  ensure_env
  prepare_workdir
  resolve_compose
  "${COMPOSE[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
}

contains_profile "$PROFILE" || die "unknown deployment profile: ${PROFILE}. Available: ${PROFILES[*]}"
COMPOSE_FILE="${DEPLOYMENTS_DIR}/${PROFILE}/podman-compose.yml"
[[ -f "$COMPOSE_FILE" ]] || die "missing compose file: ${COMPOSE_FILE}"

case "$COMMAND" in
  -h|--help|help)
    usage
    ;;
  install|up)
    doctor
    info "Starting ${PROFILE}"
    "${COMPOSE[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build
    ok "${PROFILE} started"
    ;;
  down)
    run_compose down
    ;;
  restart)
    run_compose restart
    ;;
  status)
    run_compose ps
    ;;
  logs)
    run_compose logs -f
    ;;
  doctor)
    doctor
    ;;
  config)
    ensure_env
    resolve_compose
    "${COMPOSE[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" config
    ;;
  *)
    usage >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
