#!/bin/bash
#SBATCH -J pi05_eval_mse_1246_v2
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-eval-mse-1246-v2-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-eval-mse-1246-v2-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-06:00:00

set -euo pipefail
export PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
CONDA_SH="$(find /home/S/yangrongzheng/miniconda3* /nfs_global/S/yangrongzheng/miniconda3* -maxdepth 3 -name 'conda.sh' -path '*/etc/profile.d/*' 2>/dev/null | head -1)"
source "${CONDA_SH}"
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"
export RECAP_WORKSPACE="${PI05_REPO_ROOT}/recap_workspace"
export RLINF_ROOT="${PI05_REPO_ROOT}/recap_workspace/vendor/rlinf-recap"
export RLINF_VENV_ROOT="${PI05_REPO_ROOT}/recap_workspace/.venv_recap311"
export RLINF_VENV_SITE_PACKAGES="${PI05_REPO_ROOT}/recap_workspace/.venv_recap311/lib/python3.11/site-packages"
export LIBERO_REPO_PATH="${PI05_REPO_ROOT}/recap_workspace/vendor/libero"
export CUDA_RUNTIME_ROOT="${PI05_REPO_ROOT}/recap_workspace/.venv_recap311/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
source "${PI05_REPO_ROOT}/scripts/use_local_openpi_env.sh"
cd "${PI05_REPO_ROOT}"
python3 scripts/eval_mse.py \
  --ckpt-dirs \
  results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/10000 \
  results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/20000 \
  results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/40000 \
  results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/60000 \
  --data-dir data/lerobot_pick_place \
  --max-episodes ${MAX_EPISODES:-50} \
  --num-steps ${NUM_STEPS:-10} \
  --device cuda
