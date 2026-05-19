#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-$PWD/openpi}"
CONFIG="${CONFIG:?Set CONFIG to the openpi training config name}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the trained checkpoint step directory}"

cd "$OPENPI_DIR"
export PI06_FORCE_ADVANTAGE="${PI06_FORCE_ADVANTAGE:-positive}"
uv run scripts/serve_policy.py policy:checkpoint --policy.config="$CONFIG" --policy.dir="$CHECKPOINT"

