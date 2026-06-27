#!/usr/bin/env bash
set -euo pipefail
ROOT=/nfs_global/S/yangrongzheng/pi05
VENV=$ROOT/recap_workspace/external/venv
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh || true
source "$VENV/bin/activate"
cd "$ROOT/recap_workspace/external/rlinf-recap/examples/recap/process"
python compute_returns.py --config-path "$ROOT/recap_workspace/configs" --config-name local_compute_returns "$@"
