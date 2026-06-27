#!/bin/bash
#SBATCH -J pistar06_pickplace
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pistar06-pickplace-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pistar06-pickplace-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 0-08:00:00

set -euo pipefail

export PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
export EVORL_ROOT=/nfs_global/S/yangrongzheng/evo-RL/Evo-RL
export LEROBOT_DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place_v30"
export HF_HOME="${HF_HOME:-${PI05_REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_LEROBOT_ROOT="${HF_HOME}/lerobot"
export OUTPUT_DIR="${PI05_REPO_ROOT}/results/pistar06_pickplace_$(date +%Y%m%d_%H%M%S)"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export VALUE_STEPS="${VALUE_STEPS:-8000}"
export VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-16}"
export VALUE_NUM_WORKERS="${VALUE_NUM_WORKERS:-4}"
export VALUE_DTYPE="${VALUE_DTYPE:-bfloat16}"
export VALUE_DEVICE="${VALUE_DEVICE:-cuda}"
export VALUE_LANGUAGE_REPO="${VALUE_LANGUAGE_REPO:-/nfs_global/S/yangrongzheng/pi05/models/google/gemma-3-270m}"
export VALUE_VISION_REPO="${VALUE_VISION_REPO:-/nfs_global/S/yangrongzheng/pi05/models/google/siglip-so400m-patch14-384}"
export VALUE_SAVE_FREQ="${VALUE_SAVE_FREQ:-2000}"
export VALUE_LOG_FREQ="${VALUE_LOG_FREQ:-200}"

export CAMERA_FEATURES="${CAMERA_FEATURES:-[observation.images.image,observation.images.wrist_image]}"
export SUCCESS_FIELD="${SUCCESS_FIELD:-episode_success}"
export DEFAULT_SUCCESS="${DEFAULT_SUCCESS:-success}"
export C_FAIL_COEF="${C_FAIL_COEF:-1.0}"
export STATE_FEATURE="${STATE_FEATURE:-observation.state}"
export MAX_STATE_DIM="${MAX_STATE_DIM:-32}"
export EPISODE_LABEL="${EPISODE_LABEL:-success}"

mkdir -p "${PI05_REPO_ROOT}/logs"
# Don't create OUTPUT_DIR - let lerobot do it, otherwise validate() will reject

echo "PISTAR06 pick_place value training start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "LEROBOT_DATA_DIR=${LEROBOT_DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "VALUE_STEPS=${VALUE_STEPS}"
echo "VALUE_BATCH_SIZE=${VALUE_BATCH_SIZE}"

ulimit -l unlimited || true
ulimit -v unlimited || true
ulimit -n 8192 || true

source /nfs_global/S/yangrongzheng/miniconda3-fresh/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/evo-RL/dexjoco/.conda/openpi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${EVORL_ROOT}/src:${PI05_REPO_ROOT}:${PYTHONPATH:-}"
export TMPDIR=/tmp
export MPLCONFIGDIR="/tmp/mpl-pistar06-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_DEDUP_LOGS=0
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_LEROBOT_ROOT}" "${MPLCONFIGDIR}"

echo "=== Verifying dataset ==="
if [[ ! -d "${LEROBOT_DATA_DIR}/meta" ]]; then
  echo "ERROR: Dataset meta not found at ${LEROBOT_DATA_DIR}/meta" >&2
  exit 1
fi
echo "Dataset: $(cat ${LEROBOT_DATA_DIR}/meta/info.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"total_episodes\"]} episodes, {d[\"total_frames\"]} frames')")"

echo "=== Converting episodes.jsonl to parquet format (adding episode_success) ==="
python3 "${PI05_REPO_ROOT}/scripts/convert_episodes_to_parquet.py" \
  --root "${LEROBOT_DATA_DIR}" \
  --label "${EPISODE_LABEL}"

echo "=== Starting Pistar06 value training ==="
cd "${EVORL_ROOT}"

python -m lerobot.scripts.lerobot_value_train \
  --dataset.repo_id=local/lerobot_pick_place \
  --dataset.root="${LEROBOT_DATA_DIR}" \
  --targets.success_field="${SUCCESS_FIELD}" \
  --targets.default_success="${DEFAULT_SUCCESS}" \
  --targets.c_fail_coef="${C_FAIL_COEF}" \
  --value.type=pistar06 \
  --value.dtype="${VALUE_DTYPE}" \
  --value.device="${VALUE_DEVICE}" \
  --value.vision_repo_id="${VALUE_VISION_REPO}" \
  --value.language_repo_id="${VALUE_LANGUAGE_REPO}" \
  --value.camera_features="${CAMERA_FEATURES}" \
  --value.state_feature="${STATE_FEATURE}" \
  --value.max_state_dim="${MAX_STATE_DIM}" \
  --value.normalization_mapping="{VISUAL: IDENTITY, STATE: QUANTILES, ACTION: IDENTITY}" \
  --value.use_gradient_checkpointing=true \
  --batch_size="${VALUE_BATCH_SIZE}" \
  --num_workers="${VALUE_NUM_WORKERS}" \
  --steps="${VALUE_STEPS}" \
  --save_checkpoint=true \
  --save_freq="${VALUE_SAVE_FREQ}" \
  --log_freq="${VALUE_LOG_FREQ}" \
  --wandb.enable=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=pistar06_pickplace

echo "PISTAR06 pick_place value training end: $(date '+%F %T')"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
