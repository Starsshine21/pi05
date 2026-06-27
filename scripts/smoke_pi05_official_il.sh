#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MAX_NORM_FRAMES="${MAX_NORM_FRAMES:-1024}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-assets_smoke}"
export RUN_TRAIN="${RUN_TRAIN:-0}"
export EXP_NAME="${EXP_NAME:-pickplace_pi05_lora_smoke}"

exec "$REPO_ROOT/openpi_official/run_pi05_pickplace_lora.sh" "$@"
