#!/bin/bash
#SBATCH -J pi05_pickplace_jax
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-pickplace-jax-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-pickplace-jax-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --gres-flags=enforce-binding
#SBATCH --constraint=A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH -t 3-23:00:00

set -euo pipefail

PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
OPENPI_ROOT="${PI05_REPO_ROOT}/openpi_official"
LEROBOT_DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place"
HF_LEROBOT_ROOT=${PI05_REPO_ROOT}/models/huggingface/lerobot
LEROBOT_LINK="${HF_LEROBOT_ROOT}/local/pi05-pickplace-il"
EXP_NAME="${PI05_JAX_EXP_NAME:-pi05_pickplace_jax_${SLURM_JOB_ID:-manual}}"
ASSETS_BASE_DIR="${PI05_JAX_ASSETS_BASE_DIR:-${OPENPI_ROOT}/assets}"
NORM_CONFIG_NAME="${PI05_JAX_NORM_CONFIG_NAME:-pi05_pickplace_lora}"
TRAIN_CONFIG_NAME="${PI05_JAX_TRAIN_CONFIG_NAME:-pi05_pickplace_lora}"
XLA_MEM_FRACTION="${PI05_JAX_XLA_MEM_FRACTION:-0.90}"

mkdir -p "${PI05_REPO_ROOT}/logs" "$(dirname "${LEROBOT_LINK}")"

echo "PI05 official JAX train start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "EXP_NAME=${EXP_NAME}"
echo "NORM_CONFIG_NAME=${NORM_CONFIG_NAME}"
echo "TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME}"
echo "ASSETS_BASE_DIR=${ASSETS_BASE_DIR}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"
source "${PI05_REPO_ROOT}/scripts/use_local_openpi_env.sh"

export PYTHONPATH="${OPENPI_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME="${PI05_HF_HOME:-${HF_HOME:-${PI05_REPO_ROOT}/.cache/huggingface}}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export OPENPI_DATA_HOME="${PI05_OPENPI_DATA_HOME:-${OPENPI_DATA_HOME:-${PI05_REPO_ROOT}/.cache/openpi}}"
export MPLCONFIGDIR="/tmp/mpl-openpi-${SLURM_JOB_ID:-manual}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
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

cd "${OPENPI_ROOT}"

echo "=== compute_norm_stats ==="
python scripts/compute_norm_stats.py \
  --config-name "${NORM_CONFIG_NAME}" \
  --assets-base-dir "${ASSETS_BASE_DIR}"

echo "=== train ==="
python scripts/train.py "${TRAIN_CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  --assets-base-dir "${ASSETS_BASE_DIR}" \
  --no-wandb-enabled \
  --overwrite

echo "PI05 official JAX train end: $(date '+%F %T')"
