#!/bin/bash
#SBATCH -J convert_episodes_pickplace
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/convert-episodes-pickplace-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/convert-episodes-pickplace-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --constraint=L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00

set -euo pipefail

export PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
export LEROBOT_DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place"

mkdir -p "${PI05_REPO_ROOT}/logs"

echo "=== Convert episodes.jsonl to parquet (add episode_success) ==="
echo "Start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "Data: ${LEROBOT_DATA_DIR}"

PYTHON="/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final/bin/python3"

"${PYTHON}" "${PI05_REPO_ROOT}/scripts/convert_episodes_to_parquet.py" \
  --root "${LEROBOT_DATA_DIR}" \
  --label success

echo "=== Done ==="
echo "End: $(date '+%F %T')"
