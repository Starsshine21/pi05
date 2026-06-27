#!/bin/bash
#SBATCH -J rt_recap_adj
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/rt-recap-adj-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/rt-recap-adj-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00
#SBATCH --constraint=L40S|L40

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05
ROBOTWIN_ROOT=${ROOT}/external/RoboTwin
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/data/recap_adjust_bottle_sim}
NUM_EPISODES=${NUM_EPISODES:-10}
SEED=${SEED:-0}
TASK_NAME=${TASK_NAME:-adjust_bottle}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME:-pi05_aloha_full_base}
MODEL_NAME=${MODEL_NAME:-model_robotwin}
CHECKPOINT_ID=${CHECKPOINT_ID:-30000}

mkdir -p ${ROOT}/logs

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-robotwin-recap-${SLURM_JOB_ID:-manual}}
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
echo "TASK_NAME=${TASK_NAME}"
echo "TASK_CONFIG=${TASK_CONFIG}"
echo "TRAIN_CONFIG_NAME=${TRAIN_CONFIG_NAME}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "CHECKPOINT_ID=${CHECKPOINT_ID}"
echo "NUM_EPISODES=${NUM_EPISODES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

cd ${ROOT}
python scripts/robotwin_recap_collect_sim.py \
  --task-name ${TASK_NAME} \
  --task-config ${TASK_CONFIG} \
  --train-config-name ${TRAIN_CONFIG_NAME} \
  --model-name ${MODEL_NAME} \
  --checkpoint-id ${CHECKPOINT_ID} \
  --seed ${SEED} \
  --num-episodes ${NUM_EPISODES} \
  --output-dir ${OUTPUT_DIR} \
  --overwrite

echo "Job end: $(date '+%F %T')"
echo "Raw rollout dir: ${OUTPUT_DIR}/raw_rollouts"
