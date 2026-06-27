#!/bin/bash
#SBATCH -J robotwin_pi05_demo
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/robotwin-pi05-demo-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/robotwin-pi05-demo-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH --constraint=A100

set -euo pipefail

PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
TASK_NAME=${TASK_NAME:-adjust_bottle}
GPU_ID=${GPU_ID:-0}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_aloha_full_base}
MODEL_NAME=${MODEL_NAME:-model_robotwin}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-1}

mkdir -p "${PI05_REPO_ROOT}/logs"

echo "RobotWin PI05 demo start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "TASK_NAME=${TASK_NAME}"
echo "TASK_CONFIG=${TASK_CONFIG}"
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "SEED=${SEED}"
echo "TEST_NUM=${TEST_NUM}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-robotwin-${SLURM_JOB_ID:-manual}}
export ROBOTWIN_ASSET_ID=${ROBOTWIN_ASSET_ID:-robotwin}
export PYTHONPATH="${PI05_REPO_ROOT}/external/RoboTwin:${PI05_REPO_ROOT}/external/RoboTwin/policy/pi05/src:${PI05_REPO_ROOT}/external/RoboTwin/policy/pi05/packages/openpi-client/src:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}

mkdir -p "${MPLCONFIGDIR}"

cd "${PI05_REPO_ROOT}"
bash scripts/robotwin_pi05_smoke_eval.sh "${TASK_NAME}" "${GPU_ID}" "${TASK_CONFIG}" "${TRAIN_CONFIG}" "${MODEL_NAME}" "${SEED}" "${TEST_NUM}"

echo "RobotWin PI05 demo end: $(date '+%F %T')"
