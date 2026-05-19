#!/bin/bash
#SBATCH -J pi05_pt_lora_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-pt-lora-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-pt-lora-smoke-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 0-02:00:00

set -euo pipefail

PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
OPENPI_ROOT="${PI05_REPO_ROOT}/openpi_official"
LEROBOT_DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place"
HF_LEROBOT_ROOT=/nfs_global/S/yangrongzheng/RLinf-main/models/huggingface/lerobot
LEROBOT_LINK="${HF_LEROBOT_ROOT}/local/pi05-pickplace-il"
RESULTS_DIR="${PI05_REPO_ROOT}/results"
EXP_NAME="${PI05_PT_LORA_EXP_NAME:-pi05_pt_lora_smoke_${SLURM_JOB_ID:-manual}}"

mkdir -p "${PI05_REPO_ROOT}/logs" "${RESULTS_DIR}" "$(dirname "${LEROBOT_LINK}")"

echo "PI05 PyTorch LoRA IL smoke start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

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
export OPENPI_MAX_EPISODES="${OPENPI_MAX_EPISODES:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
echo "Runtime OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
echo "Runtime HF_HOME=${HF_HOME}"
echo "Runtime HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

mkdir -p "${OPENPI_DATA_HOME}/big_vision" "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}"
cp /nfs_global/S/yangrongzheng/RLinf-main/models/openpi/big_vision/paligemma_tokenizer.model \
  "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${OPENPI_ROOT}"
python scripts/train_pytorch.py pi05_pickplace_lora_pytorch \
  --exp-name "${EXP_NAME}" \
  --checkpoint-base-dir "${RESULTS_DIR}/openpi_pytorch_lora_checkpoints" \
  --num-train-steps "${PI05_PT_LORA_STEPS:-1}" \
  --batch-size "${PI05_PT_LORA_BATCH_SIZE:-1}" \
  --num-workers "${PI05_PT_LORA_NUM_WORKERS:-0}" \
  --pytorch-training-precision "${PI05_PT_LORA_PRECISION:-float32}" \
  --log-interval 1 \
  --save-interval 1 \
  --model.pytorch-compile-mode None \
  --no-wandb-enabled \
  --overwrite

echo "PI05 PyTorch LoRA IL smoke end: $(date '+%F %T')"
