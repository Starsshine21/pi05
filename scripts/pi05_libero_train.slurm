#!/bin/bash
#SBATCH -J pi05_libero_train
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-libero-train-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-libero-train-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH -t 24:00:00
#SBATCH --constraint=L40S|A100

set -euo pipefail

PI05_REPO_ROOT="${PI05_REPO_ROOT:-/nfs_global/S/yangrongzheng/pi05}"
OPENPI_ROOT="${OPENPI_ROOT:-$PI05_REPO_ROOT/openpi_official}"
CONFIG_NAME="${PI05_LIBERO_CONFIG_NAME:-pi05_libero}"
EXP_NAME="${PI05_LIBERO_EXP_NAME:-pi05_libero_${SLURM_JOB_ID:-manual}}"
CHECKPOINT_BASE_DIR="${PI05_LIBERO_CHECKPOINT_BASE_DIR:-$PI05_REPO_ROOT/results/openpi_libero_checkpoints}"
ASSETS_BASE_DIR="${PI05_LIBERO_ASSETS_BASE_DIR:-$OPENPI_ROOT/assets}"
COMPUTE_NORM_STATS="${PI05_LIBERO_COMPUTE_NORM_STATS:-1}"
MAX_NORM_FRAMES="${PI05_LIBERO_MAX_NORM_FRAMES:-}"

mkdir -p "$PI05_REPO_ROOT/logs" "$CHECKPOINT_BASE_DIR" "$ASSETS_BASE_DIR"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "$PI05_REPO_ROOT/.conda-pi05-openpi-final"
source "$PI05_REPO_ROOT/scripts/use_pi05_libero_env.sh"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

cd "$OPENPI_ROOT"

echo "PI05 LIBERO train start: $(date '+%F %T')"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "CHECKPOINT_BASE_DIR=$CHECKPOINT_BASE_DIR"
echo "ASSETS_BASE_DIR=$ASSETS_BASE_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"

python "$PI05_REPO_ROOT/scripts/check_pi05_libero_env.py" --env-smoke

if [[ "$COMPUTE_NORM_STATS" == "1" ]]; then
  norm_args=(scripts/compute_norm_stats.py --config-name "$CONFIG_NAME" --assets-base-dir "$ASSETS_BASE_DIR")
  if [[ -n "$MAX_NORM_FRAMES" ]]; then
    norm_args+=(--max-frames "$MAX_NORM_FRAMES")
  fi
  echo "=== compute_norm_stats ==="
  python "${norm_args[@]}"
fi

echo "=== train.py ==="
python scripts/train.py "$CONFIG_NAME" \
  --exp-name "$EXP_NAME" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  --assets-base-dir "$ASSETS_BASE_DIR" \
  --overwrite

echo "PI05 LIBERO train end: $(date '+%F %T')"
