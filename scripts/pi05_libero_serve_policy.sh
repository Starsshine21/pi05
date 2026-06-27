#!/usr/bin/env bash
set -euo pipefail

PI05_REPO_ROOT="${PI05_REPO_ROOT:-/nfs_global/S/yangrongzheng/pi05}"
OPENPI_ROOT="${OPENPI_ROOT:-$PI05_REPO_ROOT/openpi_official}"
CONFIG_NAME="${PI05_LIBERO_CONFIG_NAME:-pi05_libero}"
POLICY_PORT="${PI05_LIBERO_POLICY_PORT:-8000}"
CHECKPOINT_DIR="${PI05_LIBERO_CHECKPOINT_DIR:-}"

source "$PI05_REPO_ROOT/scripts/use_pi05_libero_env.sh"
cd "$OPENPI_ROOT"

if [[ -n "$CHECKPOINT_DIR" ]]; then
  exec python scripts/serve_policy.py \
    --port "$POLICY_PORT" \
    policy:checkpoint \
    --policy.config "$CONFIG_NAME" \
    --policy.dir "$CHECKPOINT_DIR"
fi

exec python scripts/serve_policy.py --env LIBERO --port "$POLICY_PORT"
