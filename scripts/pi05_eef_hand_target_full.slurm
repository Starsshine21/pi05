#!/bin/bash
#SBATCH -J pi05_eef_hand_target
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-eef-hand-target-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-eef-hand-target-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH --constraint=L40S|A100
set -euo pipefail
ROOT=/nfs_global/S/yangrongzheng/pi05
OPENPI_ROOT="$ROOT/openpi_official"
DATASET_DIR="$ROOT/data/lerobot_pick_place_eef_hand_target"
REPO_ID="local/pi05-pickplace-eef-hand-target"
ASSETS_BASE_DIR="$OPENPI_ROOT/assets_eef_hand_target"
RUN_NAME="pi05_pickplace_eef_hand_target_train"
CHECKPOINT_BASE="$ROOT/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch"
mkdir -p "$ROOT/logs"
CONDA_SH="$(find /home/S/yangrongzheng/miniconda3* /nfs_global/S/yangrongzheng/miniconda3* -maxdepth 3 -name 'conda.sh' -path '*/etc/profile.d/*' 2>/dev/null | head -1)"
source "$CONDA_SH"
conda activate "$ROOT/.conda-pi05-openpi-final"
export RECAP_WORKSPACE="$ROOT/recap_workspace"
export RLINF_ROOT="$ROOT/recap_workspace/vendor/rlinf-recap"
export RLINF_VENV_ROOT="$ROOT/recap_workspace/.venv_recap311"
export RLINF_VENV_SITE_PACKAGES="$ROOT/recap_workspace/.venv_recap311/lib/python3.11/site-packages"
export LIBERO_REPO_PATH="$ROOT/recap_workspace/vendor/libero"
export CUDA_RUNTIME_ROOT="$ROOT/recap_workspace/.venv_recap311/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
source "$ROOT/scripts/use_local_openpi_env.sh"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$ROOT"
python scripts/convert_pick_place_to_lerobot.py \
  --input-dir /nfs_global/S/yangrongzheng/pick_place_raw_data \
  --output-dir "$DATASET_DIR" \
  --repo-id "$REPO_ID" \
  --state-mode eef_hand \
  --action-mode eef_delta \
  --use-next-state-action \
  --overwrite \
  --stride 1 \
  --image-height 224 \
  --image-width 224
cd "$OPENPI_ROOT"
python scripts/compute_norm_stats.py \
  pi05_pickplace_full_pytorch_eef_hand_target \
  --assets-base-dir "$ASSETS_BASE_DIR"
python scripts/train_pytorch.py \
  pi05_pickplace_full_pytorch_eef_hand_target \
  --exp-name "$RUN_NAME" \
  --checkpoint-base-dir "$CHECKPOINT_BASE" \
  --assets-base-dir "$ASSETS_BASE_DIR"
