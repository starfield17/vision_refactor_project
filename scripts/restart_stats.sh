#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/stop_stats.sh" "$@" || true
"${SCRIPT_DIR}/start_stats.sh" "$@"
