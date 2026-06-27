#!/bin/bash
#SBATCH -J rt_pi05_demo4
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/rt-pi05-demo4-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/rt-pi05-demo4-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-03:00:00
#SBATCH --constraint=L40S|A100

set -euo pipefail

mkdir -p /nfs_global/S/yangrongzheng/pi05/logs

ROOT=/nfs_global/S/yangrongzheng/pi05
ROBOTWIN_ROOT=${ROOT}/external/RoboTwin

GPU_ID=${GPU_ID:-0}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_aloha_full_base}
MODEL_NAME=${MODEL_NAME:-model_robotwin}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-1}

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-robotwin-pi05}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p "$MPLCONFIGDIR"

FFMPEG_SHIM_DIR=${TMPDIR}/robotwin-ffmpeg-shim
mkdir -p "$FFMPEG_SHIM_DIR"
cat > "$FFMPEG_SHIM_DIR/ffmpeg" <<'EOF'
#!/usr/bin/env bash
exec /nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 "$@"
EOF
chmod +x "$FFMPEG_SHIM_DIR/ffmpeg"
export PATH="$FFMPEG_SHIM_DIR:$PATH"

export PYTHONPATH=${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/policy/pi05/src:${ROBOTWIN_ROOT}/policy/pi05/packages/openpi-client/src:${PYTHONPATH:-}

echo "Job start: $(date '+%F %T')"
echo "hostname=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MPLCONFIGDIR=${MPLCONFIGDIR}"
echo "ffmpeg=$(command -v ffmpeg || true)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

cd ${ROOT}

python - <<'PY'
import torch
print('torch_version', torch.__version__, flush=True)
print('cuda_available', torch.cuda.is_available(), flush=True)
print('cuda_device_count', torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print('cuda_device_name', torch.cuda.get_device_name(0), flush=True)
PY

bash ${ROOT}/scripts/robotwin_pi05_record_four_demos.sh \
  ${GPU_ID} \
  ${TASK_CONFIG} \
  ${TRAIN_CONFIG} \
  ${MODEL_NAME} \
  ${SEED} \
  ${TEST_NUM}

echo "Job end: $(date '+%F %T')"
echo "Videos should be under: ${ROBOTWIN_ROOT}/eval_result"
