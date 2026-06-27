#!/bin/bash
set -euo pipefail

GPU_ID=${1:-0}
TASK_CONFIG=${2:-demo_clean}
TRAIN_CONFIG=${3:-pi05_aloha_full_base}
MODEL_NAME=${4:-model_robotwin}
SEED=${5:-0}
TEST_NUM=${6:-1}

TASKS=(
  pick_dual_bottles
  pick_diverse_bottles
  place_object_basket
  place_can_basket
)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SMOKE_SCRIPT="${ROOT_DIR}/scripts/robotwin_pi05_smoke_eval.sh"

if [ ! -x "${SMOKE_SCRIPT}" ]; then
  echo "Missing smoke eval script: ${SMOKE_SCRIPT}"
  exit 1
fi

for task in "${TASKS[@]}"; do
  echo "=================================================="
  echo "[Robotwin PI05 Demo] task=${task} gpu=${GPU_ID}"
  echo "=================================================="
  bash "${SMOKE_SCRIPT}" "${task}" "${GPU_ID}" "${TASK_CONFIG}" "${TRAIN_CONFIG}" "${MODEL_NAME}" "${SEED}" "${TEST_NUM}"
  echo
  echo "[Done] ${task}"
  echo
  sleep 1
done

echo "All requested demo tasks finished."
echo "Videos are under: ${ROOT_DIR}/external/RoboTwin/eval_result"
