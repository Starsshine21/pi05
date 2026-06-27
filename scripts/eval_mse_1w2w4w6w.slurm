#!/bin/bash
#SBATCH -J pi05_eval_mse_1246
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-eval-mse-1246-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-eval-mse-1246-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00

set -euo pipefail

export PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
export DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place"
export CHECKPOINT_BASE="${PI05_REPO_ROOT}/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train"
mkdir -p "${PI05_REPO_ROOT}/logs"

CONDA_SH="$(find /home/S/yangrongzheng/miniconda3* /nfs_global/S/yangrongzheng/miniconda3* -maxdepth 3 -name 'conda.sh' -path '*/etc/profile.d/*' 2>/dev/null | head -1)"
source "${CONDA_SH}"
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"
export RECAP_WORKSPACE="${PI05_REPO_ROOT}/recap_workspace"
export RLINF_ROOT="${PI05_REPO_ROOT}/recap_workspace/vendor/rlinf-recap"
export RLINF_VENV_ROOT="${PI05_REPO_ROOT}/recap_workspace/.venv_recap311"
export RLINF_VENV_SITE_PACKAGES="${PI05_REPO_ROOT}/recap_workspace/.venv_recap311/lib/python3.11/site-packages"
export LIBERO_REPO_PATH="${PI05_REPO_ROOT}/recap_workspace/vendor/libero"
source "${PI05_REPO_ROOT}/scripts/use_local_openpi_env.sh"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MAX_EPISODES="${MAX_EPISODES:-50}"
export NUM_STEPS="${NUM_STEPS:-10}"

python3 "${PI05_REPO_ROOT}/scripts/eval_mse.py" \
  --ckpt-dirs \
  "${CHECKPOINT_BASE}/10000" \
  "${CHECKPOINT_BASE}/20000" \
  "${CHECKPOINT_BASE}/40000" \
  "${CHECKPOINT_BASE}/60000" \
  --data-dir "${DATA_DIR}" \
  --max-episodes "${MAX_EPISODES}" \
  --num-steps "${NUM_STEPS}" \
  --device cuda
