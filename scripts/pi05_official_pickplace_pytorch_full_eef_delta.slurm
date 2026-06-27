#!/bin/bash
#SBATCH -J pi05_pickplace_pt_full_eef
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-pickplace-pt-full-eef-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-pickplace-pt-full-eef-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=L40S|A100
#SBATCH --exclude=r8l40s-a05
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH -t 1-06:00:00

set -euo pipefail

export PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
export OPENPI_ROOT="${PI05_REPO_ROOT}/openpi_official"
export LEROBOT_DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place"
export HF_HOME="${HF_HOME:-${PI05_REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_LEROBOT_ROOT="${HF_HOME}/lerobot"
export LEROBOT_LINK="${HF_LEROBOT_ROOT}/local/pi05-pickplace-il"
export RESULTS_DIR="${PI05_REPO_ROOT}/results"
export ASSETS_BASE_DIR="${OPENPI_ROOT}/assets_eef_delta"
export NORM_CONFIG_NAME="pi05_pickplace_lora"
export TRAIN_CONFIG_NAME="pi05_pickplace_full_pytorch"
export EXP_NAME="pi05_pickplace_full_pytorch_eefdelta_train"
export PALIGEMMA_TOKENIZER_SRC="${PI05_REPO_ROOT}/models/openpi/big_vision/paligemma_tokenizer.model"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((20000 + (${SLURM_JOB_ID:-0} % 20000)))"
export NNODES="${SLURM_NNODES:-1}"
export PI05_PT_EEF_STEPS="${PI05_PT_EEF_STEPS:-60000}"
export PI05_PT_EEF_SAVE_INTERVAL="${PI05_PT_EEF_SAVE_INTERVAL:-1000}"
export PI05_PT_EEF_LOG_INTERVAL="${PI05_PT_EEF_LOG_INTERVAL:-10}"
export PI05_PT_EEF_BATCH_SIZE="${PI05_PT_EEF_BATCH_SIZE:-64}"
export PI05_PT_EEF_NUM_WORKERS="${PI05_PT_EEF_NUM_WORKERS:-4}"
export PI05_PT_EEF_NORM_NUM_WORKERS="${PI05_PT_EEF_NORM_NUM_WORKERS:-4}"
export PI05_PT_EEF_PRECISION="${PI05_PT_EEF_PRECISION:-bfloat16}"
export PI05_PT_EEF_WARMUP_STEPS="${PI05_PT_EEF_WARMUP_STEPS:-10000}"
export PI05_PT_EEF_PEAK_LR="${PI05_PT_EEF_PEAK_LR:-5e-5}"
export PI05_PT_EEF_DECAY_STEPS="${PI05_PT_EEF_DECAY_STEPS:-60000}"
export PI05_PT_EEF_DECAY_LR="${PI05_PT_EEF_DECAY_LR:-1e-5}"
export PI05_PT_EEF_CLIP_GRAD_NORM="${PI05_PT_EEF_CLIP_GRAD_NORM:-1.0}"
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
NORM_STATS_PATH="${ASSETS_BASE_DIR}/pi05_pickplace/local/pi05-pickplace-il/norm_stats.json"

mkdir -p "${PI05_REPO_ROOT}/logs" "${RESULTS_DIR}" "$(dirname "${LEROBOT_LINK}")"

echo "PI05 official PyTorch FULL EEF-delta train start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "LEROBOT_DATA_DIR=${LEROBOT_DATA_DIR}"
echo "ASSETS_BASE_DIR=${ASSETS_BASE_DIR}"
echo "NORM_CONFIG_NAME=${NORM_CONFIG_NAME}"
echo "TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "NORM_STATS_PATH=${NORM_STATS_PATH}"

ulimit -l unlimited
ulimit -v unlimited

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${OPENPI_ROOT}/src:${PI05_REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR=/tmp
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${PI05_REPO_ROOT}/.cache/openpi}"
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

if [[ ! -d "${LEROBOT_DATA_DIR}" ]]; then
  echo "Missing LeRobot dataset dir: ${LEROBOT_DATA_DIR}" >&2
  exit 1
fi

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
if [[ -f "${NORM_STATS_PATH}" ]]; then
  echo "Found existing norm stats at ${NORM_STATS_PATH}; skipping recompute."
else
  NORM_CMD=(python scripts/compute_norm_stats.py
    --config-name "${NORM_CONFIG_NAME}"
    --assets-base-dir "${ASSETS_BASE_DIR}"
    --num-workers "${PI05_PT_EEF_NORM_NUM_WORKERS}")
  printf 'Running norm command:'
  printf ' %q' "${NORM_CMD[@]}"
  printf '\n'
  "${NORM_CMD[@]}"
fi

echo "=== train_pytorch ==="
srun --export=ALL --ntasks=${WORLD_SIZE} --ntasks-per-node=${GPUS_PER_NODE} bash -lc '
  source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
  conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${OPENPI_ROOT}/src:${PI05_REPO_ROOT}:${PYTHONPATH:-}"
  export RANK=${SLURM_PROCID}
  export WORLD_SIZE='"${WORLD_SIZE}"'
  export LOCAL_RANK=${SLURM_LOCALID}
  export MASTER_ADDR='"${MASTER_ADDR}"'
  export MASTER_PORT='"${MASTER_PORT}"'
  cd '"${OPENPI_ROOT}"'
  python scripts/train_pytorch.py '"${TRAIN_CONFIG_NAME}"' \
    --exp_name '"${EXP_NAME}"' \
    --checkpoint-base-dir '"${RESULTS_DIR}/openpi_official_pytorch_full_checkpoints"' \
    --assets-base-dir '"${ASSETS_BASE_DIR}"' \
    --num_train_steps '"${PI05_PT_EEF_STEPS}"' \
    --batch_size '"${PI05_PT_EEF_BATCH_SIZE}"' \
    --num_workers '"${PI05_PT_EEF_NUM_WORKERS}"' \
    --pytorch_training_precision '"${PI05_PT_EEF_PRECISION}"' \
    --log_interval '"${PI05_PT_EEF_LOG_INTERVAL}"' \
    --save_interval '"${PI05_PT_EEF_SAVE_INTERVAL}"' \
    --lr_schedule.warmup_steps '"${PI05_PT_EEF_WARMUP_STEPS}"' \
    --lr_schedule.peak_lr '"${PI05_PT_EEF_PEAK_LR}"' \
    --lr_schedule.decay_steps '"${PI05_PT_EEF_DECAY_STEPS}"' \
    --lr_schedule.decay_lr '"${PI05_PT_EEF_DECAY_LR}"' \
    --optimizer.clip_gradient_norm '"${PI05_PT_EEF_CLIP_GRAD_NORM}"' \
    --no-wandb-enabled \
    --resume
'

echo "PI05 official PyTorch FULL EEF-delta train end: $(date '+%F %T')"
