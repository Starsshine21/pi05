#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-$PWD/openpi}"
CONFIG="${CONFIG:-pi05_libero}"
EXP_NAME="${EXP_NAME:-pi06_recap_run}"
ADVANTAGE_JSONL="${ADVANTAGE_JSONL:?Set ADVANTAGE_JSONL to the JSONL produced by make_advantage_labels.py}"

cd "$OPENPI_DIR"
export PI06_ADVANTAGE_JSONL="$ADVANTAGE_JSONL"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

uv run scripts/compute_norm_stats.py --config-name "$CONFIG"
uv run scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" --overwrite

echo "Checkpoint directory: $OPENPI_DIR/checkpoints/$CONFIG/$EXP_NAME"

