#!/usr/bin/env bash
set -euo pipefail

PI05_REPO_ROOT="${PI05_REPO_ROOT:-/nfs_global/S/yangrongzheng/pi05}"
OPENPI_ROOT="${OPENPI_ROOT:-$PI05_REPO_ROOT/openpi_official}"
HOST="${PI05_LIBERO_POLICY_HOST:-127.0.0.1}"
PORT="${PI05_LIBERO_POLICY_PORT:-8000}"
TASK_SUITE="${PI05_LIBERO_TASK_SUITE:-libero_10}"
TRIALS="${PI05_LIBERO_TRIALS:-10}"
VIDEO_OUT_PATH="${PI05_LIBERO_VIDEO_OUT_PATH:-$PI05_REPO_ROOT/results/libero/videos}"

source "$PI05_REPO_ROOT/scripts/use_pi05_libero_env.sh"
mkdir -p "$VIDEO_OUT_PATH"
cd "$OPENPI_ROOT"

exec python examples/libero/main.py \
  --args.host "$HOST" \
  --args.port "$PORT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task "$TRIALS" \
  --args.video-out-path "$VIDEO_OUT_PATH"
