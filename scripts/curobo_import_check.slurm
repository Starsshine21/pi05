#!/bin/bash
#SBATCH -J curobo_chk
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/curobo-check-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/curobo-check-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-00:20:00
#SBATCH --constraint=L40S|A100

set -euo pipefail
mkdir -p /nfs_global/S/yangrongzheng/pi05/logs
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=8.9
cd /nfs_global/S/yangrongzheng/pi05
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
import curobo
print('curobo', curobo.__file__)
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
print('curobo core imports ok')
PY
