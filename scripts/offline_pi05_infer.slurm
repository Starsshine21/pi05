#!/bin/bash
#SBATCH -J pi05_offline_infer
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-offline-infer-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-offline-infer-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0-00:30:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05
mkdir -p "$ROOT/logs" "$ROOT/analysis"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-$ROOT/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/19000}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_pickplace_full_pytorch}
EPISODE_INDEX=${EPISODE_INDEX:-0}
FRAME_INDEX=${FRAME_INDEX:-0}
DATA_DIR=${DATA_DIR:-$ROOT/data/lerobot_pick_place}
OUTPUT_JSON=${OUTPUT_JSON:-$ROOT/analysis/offline_infer_ep${EPISODE_INDEX}_fr${FRAME_INDEX}.json}

CONDA_SH="$(find /home/S/yangrongzheng/miniconda3* /nfs_global/S/yangrongzheng/miniconda3* -maxdepth 3 -name 'conda.sh' -path '*/etc/profile.d/*' 2>/dev/null | head -1)"
source "$CONDA_SH"
conda activate "$ROOT/.conda-pi05-openpi-final"
export RECAP_WORKSPACE="$ROOT/recap_workspace"
export RLINF_ROOT="$ROOT/recap_workspace/vendor/rlinf-recap"
export RLINF_VENV_ROOT="$ROOT/recap_workspace/.venv_recap311"
export RLINF_VENV_SITE_PACKAGES="$ROOT/recap_workspace/.venv_recap311/lib/python3.11/site-packages"
export LIBERO_REPO_PATH="$ROOT/recap_workspace/vendor/libero"
source "$ROOT/scripts/use_local_openpi_env.sh"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$ROOT"
python "$ROOT/scripts/offline_pi05_infer.py" \
  --data-dir "$DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --train-config "$TRAIN_CONFIG" \
  --episode-index "$EPISODE_INDEX" \
  --frame-index "$FRAME_INDEX" \
  --output-json "$OUTPUT_JSON"
