#!/bin/bash
#SBATCH -J pi05_pickplace_pt_full_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-pickplace-pt-full-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-pickplace-pt-full-smoke-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH -t 00:30:00
#SBATCH --constraint=L40S|A100

set -euo pipefail

PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
OPENPI_ROOT="${PI05_REPO_ROOT}/openpi_official"
LEROBOT_DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place"
HF_LEROBOT_ROOT=${PI05_REPO_ROOT}/models/huggingface/lerobot
LEROBOT_LINK="${HF_LEROBOT_ROOT}/local/pi05-pickplace-il"
RESULTS_DIR="${PI05_REPO_ROOT}/results"
ASSETS_BASE_DIR="${PI05_PT_FULL_ASSETS_BASE_DIR:-${OPENPI_ROOT}/assets_smoke}"
NORM_CONFIG_NAME="${PI05_PT_FULL_NORM_CONFIG_NAME:-pi05_pickplace_lora}"
TRAIN_CONFIG_NAME="${PI05_PT_FULL_TRAIN_CONFIG_NAME:-pi05_pickplace_full_pytorch}"
EXP_NAME="${PI05_PT_FULL_EXP_NAME:-pi05_pickplace_full_smoke_${SLURM_JOB_ID:-manual}}"
PALIGEMMA_TOKENIZER_SRC=${PI05_REPO_ROOT}/models/openpi/big_vision/paligemma_tokenizer.model
MASTER_ADDR="$(hostname)"
MASTER_PORT="${PI05_PT_FULL_MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"

mkdir -p "${PI05_REPO_ROOT}/logs" "${RESULTS_DIR}" "$(dirname "${LEROBOT_LINK}")"

echo "PI05 official PyTorch FULL smoke start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NORM_CONFIG_NAME=${NORM_CONFIG_NAME}"
echo "TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME}"
echo "EXP_NAME=${EXP_NAME}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"
source "${PI05_REPO_ROOT}/scripts/use_local_openpi_env.sh"

export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export TMPDIR=/tmp
export OPENPI_DATA_HOME="${PI05_OPENPI_DATA_HOME:-${OPENPI_DATA_HOME:-${PI05_REPO_ROOT}/.cache/openpi}}"
export HF_HOME="${PI05_HF_HOME:-${HF_HOME:-${PI05_REPO_ROOT}/.cache/huggingface}}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MPLCONFIGDIR="/tmp/mpl-openpi-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p "${HF_DATASETS_CACHE}" "${OPENPI_DATA_HOME}" "${MPLCONFIGDIR}" "${ASSETS_BASE_DIR}"

if [[ -L "${LEROBOT_LINK}" ]]; then
  current_target="$(readlink "${LEROBOT_LINK}")"
  if [[ "${current_target}" != "${LEROBOT_DATA_DIR}" ]]; then
    rm -f "${LEROBOT_LINK}"
    ln -s "${LEROBOT_DATA_DIR}" "${LEROBOT_LINK}"
  fi
elif [[ -e "${LEROBOT_LINK}" ]]; then
  echo "${LEROBOT_LINK} exists and is not a symlink; refusing to replace it." >&2
  exit 1
else
  ln -s "${LEROBOT_DATA_DIR}" "${LEROBOT_LINK}"
fi

if [[ ! -s "${PALIGEMMA_TOKENIZER_SRC}" ]]; then
  echo "Missing tokenizer seed file: ${PALIGEMMA_TOKENIZER_SRC}" >&2
  exit 1
fi
mkdir -p "${OPENPI_DATA_HOME}/big_vision"
cp "${PALIGEMMA_TOKENIZER_SRC}" "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
touch "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model.ready"

cd "${OPENPI_ROOT}"

echo "=== compute_norm_stats ==="
python scripts/compute_norm_stats.py \
  --config-name "${NORM_CONFIG_NAME}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --max-frames 1 \
  --num-workers 0

echo "=== train_pytorch ==="
python scripts/train_pytorch.py "${TRAIN_CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --checkpoint-base-dir "${RESULTS_DIR}/openpi_official_pytorch_full_smoke_checkpoints" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --num_train_steps 1 \
  --batch_size 1 \
  --num_workers 0 \
  --pytorch_training_precision "${PI05_PT_FULL_PRECISION:-bfloat16}" \
  --log_interval 1 \
  --save_interval 1 \
  --no-wandb-enabled \
  --overwrite

echo "PI05 official PyTorch FULL smoke end: $(date '+%F %T')"
