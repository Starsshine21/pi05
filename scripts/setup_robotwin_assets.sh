#!/usr/bin/env bash
set -euo pipefail

PI05_REPO_ROOT="${PI05_REPO_ROOT:-/nfs_global/S/yangrongzheng/pi05}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$PI05_REPO_ROOT/external/RoboTwin}"
ASSETS_ROOT="${ROBOTWIN_ASSETS_ROOT:-$PI05_REPO_ROOT/../assets}"

if [[ ! -d "$ROBOTWIN_ROOT" ]]; then
  echo "Missing RoboTwin repo at $ROBOTWIN_ROOT" >&2
  exit 1
fi

if [[ ! -d "$ASSETS_ROOT/embodiments" || ! -d "$ASSETS_ROOT/objects" ]]; then
  echo "Missing RoboTwin assets under $ASSETS_ROOT" >&2
  exit 1
fi

cd "$ROBOTWIN_ROOT"
export ASSETS_PATH="$(cd "$PI05_REPO_ROOT/.." && pwd)"
printf 'y\n%s\n' "$ASSETS_PATH" | python script/update_embodiment_config_path.py

echo "ASSETS_PATH=$ASSETS_PATH"
echo "RoboTwin assets configured."
