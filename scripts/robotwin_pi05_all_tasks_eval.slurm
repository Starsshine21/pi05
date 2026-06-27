#!/bin/bash
#SBATCH -J rt_pi05_all_eval
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/rt-pi05-all-eval-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/rt-pi05-all-eval-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1-04:00:00
#SBATCH --constraint=L40S|L40

set -euo pipefail

mkdir -p /nfs_global/S/yangrongzheng/pi05/logs

ROOT=/nfs_global/S/yangrongzheng/pi05
ROBOTWIN_ROOT=${ROOT}/external/RoboTwin
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_aloha_full_base}
MODEL_NAME=${MODEL_NAME:-model_robotwin}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-50}
TASK_FILTER=${TASK_FILTER:-}

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-robotwin-pi05-${SLURM_JOB_ID:-manual}}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$MPLCONFIGDIR"

FFMPEG_SHIM_DIR=${TMPDIR}/robotwin-ffmpeg-shim-${SLURM_JOB_ID:-manual}
mkdir -p "$FFMPEG_SHIM_DIR"
cat > "$FFMPEG_SHIM_DIR/ffmpeg" <<'EOS'
#!/usr/bin/env bash
exec /nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 "$@"
EOS
chmod +x "$FFMPEG_SHIM_DIR/ffmpeg"
export PATH="$FFMPEG_SHIM_DIR:$PATH"

export PYTHONPATH=${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/policy/pi05/src:${ROBOTWIN_ROOT}/policy/pi05/packages/openpi-client/src:${PYTHONPATH:-}

echo "Job start: $(date '+%F %T')"
echo "hostname=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "TASK_CONFIG=${TASK_CONFIG}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SEED=${SEED}"
echo "TEST_NUM=${TEST_NUM}"
echo "TASK_FILTER=${TASK_FILTER}"
echo "ffmpeg=$(command -v ffmpeg || true)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

cd ${ROBOTWIN_ROOT}

mapfile -t TASKS < <(
  find envs -maxdepth 1 -name '*.py' -printf '%f\n' \
    | sed 's/\.py$//' \
    | grep -v '^_' \
    | grep -v '^__init__$' \
    | sort
)

if [[ -n "${TASK_FILTER}" ]]; then
  mapfile -t TASKS < <(printf '%s\n' "${TASKS[@]}" | grep -E "${TASK_FILTER}")
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "No tasks matched."
  exit 1
fi

echo "Total tasks: ${#TASKS[@]}"
printf '%s\n' "${TASKS[@]}"

declare -a SUMMARY_LINES=()

run_eval_task() {
  local task_name="$1"
  PYTHONWARNINGS=ignore::UserWarning \
  python script/eval_policy.py --config policy/pi05/deploy_policy.yml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${TASK_CONFIG}" \
    --train_config_name "${TRAIN_CONFIG}" \
    --model_name "${MODEL_NAME}" \
    --ckpt_setting "${MODEL_NAME}" \
    --seed "${SEED}" \
    --policy_name pi05 \
    --test_num "${TEST_NUM}"
}

for TASK_NAME in "${TASKS[@]}"; do
  echo
  echo "===== TASK ${TASK_NAME} START $(date '+%F %T') ====="

  BEFORE_LATEST=$(find "eval_result/${TASK_NAME}/pi05/${TASK_CONFIG}/${MODEL_NAME}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)

  TASK_STATUS="OK"
  if ! run_eval_task "${TASK_NAME}"; then
    TASK_STATUS="FAILED"
    echo "TASK=${TASK_NAME} STATUS=FAILED"
  fi

  AFTER_LATEST=$(find "eval_result/${TASK_NAME}/pi05/${TASK_CONFIG}/${MODEL_NAME}" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)
  RESULT_FILE=""
  if [[ -n "${AFTER_LATEST}" && "${AFTER_LATEST}" != "${BEFORE_LATEST}" && -f "${AFTER_LATEST}/_result.txt" ]]; then
    RESULT_FILE="${AFTER_LATEST}/_result.txt"
  elif [[ -n "${AFTER_LATEST}" && -f "${AFTER_LATEST}/_result.txt" ]]; then
    RESULT_FILE="${AFTER_LATEST}/_result.txt"
  fi

  if [[ -n "${RESULT_FILE}" ]]; then
    SUCCESS_RATE=$(tail -n 1 "${RESULT_FILE}")
    printf 'TASK=%s STATUS=%s SUCCESS_RATE=%s RESULT=%s\n' "${TASK_NAME}" "${TASK_STATUS}" "${SUCCESS_RATE}" "${RESULT_FILE}"
    SUMMARY_LINES+=("TASK=${TASK_NAME} STATUS=${TASK_STATUS} SUCCESS_RATE=${SUCCESS_RATE} RESULT=${RESULT_FILE}")
  else
    printf 'TASK=%s STATUS=%s SUCCESS_RATE=UNKNOWN RESULT=missing\n' "${TASK_NAME}" "${TASK_STATUS}"
    SUMMARY_LINES+=("TASK=${TASK_NAME} STATUS=${TASK_STATUS} SUCCESS_RATE=UNKNOWN RESULT=missing")
  fi

  echo "===== TASK ${TASK_NAME} END $(date '+%F %T') ====="
done

echo
echo "===== SUMMARY START ====="
printf '%s\n' "${SUMMARY_LINES[@]}"
echo "===== SUMMARY END ====="
echo "Job end: $(date '+%F %T')"
