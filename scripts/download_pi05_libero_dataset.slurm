#!/bin/bash
#SBATCH -J dpresnet_pp
#SBATCH -o ret-%j.out
#SBATCH -e ret-%j.err

#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH -t 3-23:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --gres-flags=enforce-binding


set -euo pipefail

PI05_REPO_ROOT="${PI05_REPO_ROOT:-/nfs_global/S/yangrongzheng/pi05}"
DATASET_REPO_ID="${PI05_LIBERO_DATASET_REPO_ID:-physical-intelligence/libero}"
DATASET_LOCAL_DIR="${PI05_LIBERO_DATASET_LOCAL_DIR:-${PI05_REPO_ROOT}/data/hf_datasets/physical-intelligence/libero}"
RETRIES="${PI05_LIBERO_DOWNLOAD_RETRIES:-100}"
SLEEP_SECONDS="${PI05_LIBERO_DOWNLOAD_SLEEP:-10}"
INCLUDE_PREFIX="${PI05_LIBERO_DOWNLOAD_INCLUDE_PREFIX:-}"

mkdir -p "${PI05_REPO_ROOT}/logs" "${DATASET_LOCAL_DIR}"

echo "PI05 LIBERO dataset download start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "PI05_REPO_ROOT=${PI05_REPO_ROOT}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "DATASET_LOCAL_DIR=${DATASET_LOCAL_DIR}"
echo "RETRIES=${RETRIES}"
echo "SLEEP_SECONDS=${SLEEP_SECONDS}"
echo "INCLUDE_PREFIX=${INCLUDE_PREFIX}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"

# Do not inherit old RLinf/libero/proxy settings from the submitting shell.
# The failed 758455 run inherited a dead SOCKS proxy on the compute node and
# routed HuggingFace requests through /RLinf-main/.venv's httpx stack.
unset PYTHONPATH
unset LIBERO_REPO_PATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy

export OPENPI_ROOT="${PI05_REPO_ROOT}/openpi_official"
export LIBERO_REPO_PATH="${OPENPI_ROOT}/third_party/libero"
source "${PI05_REPO_ROOT}/scripts/use_pi05_libero_env.sh"

# Direct HuggingFace worked better than hf-mirror on this cluster, but long
# object streams were unstable. Keep transfers conservative and let the Python
# wrapper retry each file independently.
unset HF_ENDPOINT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
unset http_proxy https_proxy all_proxy no_proxy
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"

ARGS=(
  "${DATASET_REPO_ID}"
  --repo-type dataset
  --local-dir "${DATASET_LOCAL_DIR}"
  --retries "${RETRIES}"
  --sleep "${SLEEP_SECONDS}"
)

if [[ -n "${INCLUDE_PREFIX}" ]]; then
  ARGS+=(--include-prefix "${INCLUDE_PREFIX}")
fi

python "${PI05_REPO_ROOT}/scripts/download_hf_dataset_retry.py" "${ARGS[@]}"

echo "=== download summary ==="
du -sh "${DATASET_LOCAL_DIR}" || true
find "${DATASET_LOCAL_DIR}" -type f -name '*.parquet' | wc -l
find "${DATASET_LOCAL_DIR}/meta" -maxdepth 1 -type f -printf '%p %s\n' 2>/dev/null | sort || true

echo "PI05 LIBERO dataset download end: $(date '+%F %T')"
