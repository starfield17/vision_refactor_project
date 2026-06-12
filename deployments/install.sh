#!/usr/bin/env bash
set -euo pipefail

DEPLOYMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${DEPLOYMENTS_DIR}/.." && pwd)"
PROFILE="${1:-all-in-one}"
COMPOSE_FILE="${DEPLOYMENTS_DIR}/${PROFILE}/podman-compose.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "[ERROR] Unknown deployment profile: ${PROFILE}" >&2
  echo "Available: all-in-one, edge, remote-gpu" >&2
  exit 2
fi

if ! command -v podman >/dev/null 2>&1; then
  echo "[ERROR] podman is required" >&2
  exit 2
fi

if podman compose version >/dev/null 2>&1; then
  COMPOSE=(podman compose)
elif command -v podman-compose >/dev/null 2>&1; then
  COMPOSE=(podman-compose)
else
  echo "[ERROR] podman compose or podman-compose is required" >&2
  exit 2
fi

ENV_FILE="${DEPLOYMENTS_DIR}/${PROFILE}/.env"
if [[ ! -f "$ENV_FILE" && -f "${DEPLOYMENTS_DIR}/${PROFILE}/.env.example" ]]; then
  cp "${DEPLOYMENTS_DIR}/${PROFILE}/.env.example" "$ENV_FILE"
  echo "[INFO] Created ${ENV_FILE}"
fi

mkdir -p "${PROJECT_DIR}/work-dir"/{artifacts,state,stats,tmp,models,datasets,runs,outputs}

echo "[INFO] Starting ${PROFILE} with ${COMPOSE[*]}"
"${COMPOSE[@]}" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --build

echo "[OK] ${PROFILE} started"
echo "Logs: ${COMPOSE[*]} -f ${COMPOSE_FILE} logs -f"
