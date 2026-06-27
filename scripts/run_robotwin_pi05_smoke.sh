#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp}
export PYTHONPATH="${ROOT_DIR}/external/RoboTwin:${ROOT_DIR}/external/RoboTwin/policy/pi05/src:${ROOT_DIR}/external/RoboTwin/policy/pi05/packages/openpi-client/src:${PYTHONPATH:-}"

cd "${ROOT_DIR}"
python scripts/check_robotwin_pi05_smoke.py "$@"
