#!/bin/bash
#SBATCH -J rt_pi05_adj1
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/rt-pi05-adj1-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/rt-pi05-adj1-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-01:30:00
#SBATCH --constraint=L40S|L40

set -euo pipefail

mkdir -p /nfs_global/S/yangrongzheng/pi05/logs

ROOT=/nfs_global/S/yangrongzheng/pi05
ROBOTWIN_ROOT=${ROOT}/external/RoboTwin
TASK_NAME=${TASK_NAME:-adjust_bottle}
GPU_ID=${GPU_ID:-0}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_aloha_full_base}
MODEL_NAME=${MODEL_NAME:-model_robotwin}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-100}

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-robotwin-pi05}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$MPLCONFIGDIR"

FFMPEG_SHIM_DIR=${TMPDIR}/robotwin-ffmpeg-shim
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
echo "MPLCONFIGDIR=${MPLCONFIGDIR}"
echo "ffmpeg=$(command -v ffmpeg || true)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

cd ${ROBOTWIN_ROOT}
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/pi05/deploy_policy.yml \
  --overrides \
  --task_name ${TASK_NAME} \
  --task_config ${TASK_CONFIG} \
  --train_config_name ${TRAIN_CONFIG} \
  --model_name ${MODEL_NAME} \
  --ckpt_setting ${MODEL_NAME} \
  --seed ${SEED} \
  --policy_name pi05 \
  --test_num ${TEST_NUM}

echo "Job end: $(date '+%F %T')"
echo "Results under: ${ROBOTWIN_ROOT}/eval_result/${TASK_NAME}/pi05/${TASK_CONFIG}/${MODEL_NAME}"
