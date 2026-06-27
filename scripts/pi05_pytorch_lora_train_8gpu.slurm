#!/bin/bash
#SBATCH -J pi05_pt_lora_8g
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-pt-lora-8g-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-pt-lora-8g-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH -t 0-04:00:00

set -euo pipefail

PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
OPENPI_ROOT="${PI05_REPO_ROOT}/openpi_official"
LEROBOT_DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place"
HF_LEROBOT_ROOT=/models/huggingface/lerobot
LEROBOT_LINK="${HF_LEROBOT_ROOT}/local/pi05-pickplace-il"
RESULTS_DIR="${PI05_REPO_ROOT}/results"
EXP_NAME="${PI05_PT_LORA_EXP_NAME:-pi05_pt_lora_8g_${SLURM_JOB_ID:-manual}}"
PALIGEMMA_TOKENIZER_SRC=/models/openpi/big_vision/paligemma_tokenizer.model
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${PI05_PT_LORA_MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"
NNODES="${SLURM_NNODES:-2}"
GPUS_PER_NODE=4
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

mkdir -p "${PI05_REPO_ROOT}/logs" "${RESULTS_DIR}" "$(dirname "${LEROBOT_LINK}")"

echo "PI05 PyTorch LoRA IL 8GPU train start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES} WORLD_SIZE=${WORLD_SIZE}"

ulimit -l unlimited
ulimit -v unlimited

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"
source "${PI05_REPO_ROOT}/scripts/use_local_openpi_env.sh"

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
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
echo "Runtime OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "Runtime HF_HOME=${HF_HOME}"
echo "Runtime HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

if [[ ! -s "${PALIGEMMA_TOKENIZER_SRC}" ]]; then
  echo "Missing tokenizer seed file: ${PALIGEMMA_TOKENIZER_SRC}" >&2
  exit 1
fi

mkdir -p "${OPENPI_DATA_HOME}/big_vision" "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}"
cp "${PALIGEMMA_TOKENIZER_SRC}" "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
touch "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model.ready"
export PALIGEMMA_TOKENIZER_SRC

srun --export=ALL --ntasks=${WORLD_SIZE} --ntasks-per-node=${GPUS_PER_NODE} bash -lc '
  source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
  conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
  source /nfs_global/S/yangrongzheng/pi05/scripts/use_local_openpi_env.sh >/dev/null
  tokenizer_dst="${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
  tokenizer_ready="${tokenizer_dst}.ready"
  mkdir -p "$(dirname "${tokenizer_dst}")" "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}"
  if [[ "${SLURM_LOCALID}" == "0" ]]; then
    if [[ ! -s "${tokenizer_dst}" ]]; then
      rm -f "${tokenizer_ready}"
      tokenizer_tmp="${tokenizer_dst}.tmp.${SLURM_JOB_ID}.${SLURM_NODEID:-0}"
      cp "${PALIGEMMA_TOKENIZER_SRC}" "${tokenizer_tmp}"
      mv "${tokenizer_tmp}" "${tokenizer_dst}"
    fi
    touch "${tokenizer_ready}"
  fi
  while [[ ! -s "${tokenizer_dst}" || ! -e "${tokenizer_ready}" ]]; do
    sleep 1
  done
  echo "=== $(hostname) rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} ==="
  export RANK=${SLURM_PROCID}
  export WORLD_SIZE='"${WORLD_SIZE}"'
  export LOCAL_RANK=${SLURM_LOCALID}
  export MASTER_ADDR='"${MASTER_ADDR}"'
  export MASTER_PORT='"${MASTER_PORT}"'
  python - <<"PY"
import sys
print("python", sys.executable, flush=True)
import jax
print("jax_ok", jax.__file__, flush=True)
PY
  cd '"${OPENPI_ROOT}"'
  python scripts/train_pytorch.py pi05_pickplace_lora_pytorch \
    --exp_name '"${EXP_NAME}"' \
    --checkpoint-base-dir '"${RESULTS_DIR}/openpi_pytorch_lora_checkpoints"' \
    --num_train_steps '"${PI05_PT_LORA_STEPS:-20000}"' \
    --batch_size '"${PI05_PT_LORA_BATCH_SIZE:-32}"' \
    --num_workers '"${PI05_PT_LORA_NUM_WORKERS:-2}"' \
    --pytorch_training_precision '"${PI05_PT_LORA_PRECISION:-bfloat16}"' \
    --log_interval '"${PI05_PT_LORA_LOG_INTERVAL:-10}"' \
    --save_interval '"${PI05_PT_LORA_SAVE_INTERVAL:-1000}"' \
    --no-wandb-enabled \
    --overwrite
'

echo "PI05 PyTorch LoRA IL 8GPU train end: $(date '+%F %T')"
