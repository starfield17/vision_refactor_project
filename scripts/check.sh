#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

PY_TARGETS=(
  common
  core
  control_plane
  train
  autolabel
  edge_agent
  remote_worker
  stats_service
  tests
  scripts
)

COMPILE_TARGETS=(
  common
  core
  control_plane
  train
  autolabel
  edge_agent
  remote_worker
  stats_service
)

usage() {
  cat <<'EOF'
Usage: bash scripts/check.sh [--fix]

Runs the local verification suite. By default the script is check-only.
Use --fix to run Ruff auto-fixes and formatting before verification.
EOF
}

FIX=0
for arg in "$@"; do
  case "$arg" in
    --fix)
      FIX=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$FIX" -eq 1 ]]; then
  "$PYTHON_BIN" -m ruff check --fix "${PY_TARGETS[@]}"
  "$PYTHON_BIN" -m ruff format "${PY_TARGETS[@]}"
fi

PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m ruff check "${PY_TARGETS[@]}"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m ruff format --check "${PY_TARGETS[@]}"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m pytest -q
PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m compileall -q "${COMPILE_TARGETS[@]}"
"$PYTHON_BIN" -m pip wheel . --no-deps -w /tmp/vision_wheel_test
npm --prefix control_plane/web run build
bash -n deployments/install.sh
podman-compose -f deployments/all-in-one/podman-compose.yml config >/dev/null
podman-compose -f deployments/edge/podman-compose.yml config >/dev/null
podman-compose -f deployments/remote-gpu/podman-compose.yml config >/dev/null
