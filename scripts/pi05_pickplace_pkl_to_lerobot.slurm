#!/bin/bash
#SBATCH -J pi05_pkl2lerobot
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-pkl2lerobot-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-pkl2lerobot-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-12:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05
INPUT_DIR="${PI05_PICKPLACE_PKL_INPUT_DIR:-/nfs_global/S/yangrongzheng/pick_place_raw_data}"
OUTPUT_DIR="${PI05_PICKPLACE_LEROBOT_OUTPUT_DIR:-$ROOT/data/lerobot_pick_place}"
REPO_ID="${PI05_PICKPLACE_LEROBOT_REPO_ID:-local/pi05-pickplace-il}"
ACTION_MODE="${PI05_PICKPLACE_ACTION_MODE:-eef_delta}"
USE_NEXT_STATE_ACTION="${PI05_PICKPLACE_USE_NEXT_STATE_ACTION:-0}"
IMAGE_HEIGHT="${PI05_PICKPLACE_IMAGE_HEIGHT:-224}"
IMAGE_WIDTH="${PI05_PICKPLACE_IMAGE_WIDTH:-224}"
MAX_EPISODES="${PI05_PICKPLACE_MAX_EPISODES:-}"
STRIDE="${PI05_PICKPLACE_STRIDE:-1}"

mkdir -p "$ROOT/logs" "$ROOT/data"

echo "Job start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "REPO_ID=${REPO_ID}"
echo "ACTION_MODE=${ACTION_MODE}"
echo "USE_NEXT_STATE_ACTION=${USE_NEXT_STATE_ACTION}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "$ROOT/.conda-pi05-openpi-final"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH:-}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "PATH=$PATH"
echo "PYTHONPATH=$PYTHONPATH"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

CMD=(python "$ROOT/scripts/convert_pick_place_to_lerobot.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --repo-id "$REPO_ID" \
  --action-mode "$ACTION_MODE" \
  --stride "$STRIDE" \
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
echo "Summary file: ${OUTPUT_DIR}/conversion_summary.json"
