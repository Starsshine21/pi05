#!/bin/bash
#SBATCH -J pi05_pkl2zarr
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-pkl2zarr-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-pkl2zarr-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05
INPUT_DIR="${PI05_PICKPLACE_PKL_INPUT_DIR:-/nfs_global/S/yangrongzheng/pick_place_raw_data}"
OUTPUT_PATH="${PI05_PICKPLACE_ZARR_OUTPUT_PATH:-$ROOT/data/zarr_pick_place_eef_delta.zarr}"
IMAGE_HEIGHT="${PI05_PICKPLACE_IMAGE_HEIGHT:-224}"
IMAGE_WIDTH="${PI05_PICKPLACE_IMAGE_WIDTH:-224}"
MAX_EPISODES="${PI05_PICKPLACE_MAX_EPISODES:-}"
USE_NEXT_STATE_ACTION="${PI05_PICKPLACE_USE_NEXT_STATE_ACTION:-0}"

mkdir -p "$ROOT/logs" "$ROOT/data"

echo "Job start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "IMAGE_HEIGHT=${IMAGE_HEIGHT}"
echo "IMAGE_WIDTH=${IMAGE_WIDTH}"
echo "MAX_EPISODES=${MAX_EPISODES:-ALL}"
echo "USE_NEXT_STATE_ACTION=${USE_NEXT_STATE_ACTION}"

ulimit -l unlimited
ulimit -v unlimited

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/.conda-pi05-openpi-final"
source "$ROOT/scripts/use_local_openpi_env.sh"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/openpi_official/src:${PYTHONPATH:-}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

CMD=(python "$ROOT/scripts/convert_pick_place_pkl_to_zarr.py" \
  --input-dir "$INPUT_DIR" \
  --output-path "$OUTPUT_PATH" \
  --image-height "$IMAGE_HEIGHT" \
  --image-width "$IMAGE_WIDTH" \
  --overwrite)

if [[ -n "$MAX_EPISODES" ]]; then
  CMD+=(--max-episodes "$MAX_EPISODES")
fi

if [[ "$USE_NEXT_STATE_ACTION" == "1" ]]; then
  CMD+=(--use-next-state-action)
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "Job end: $(date '+%F %T')"
echo "Summary file: ${OUTPUT_PATH}/conversion_summary.json"
