#!/bin/bash
#SBATCH -J p3d_ops
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/p3d-ops-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/p3d-ops-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-00:20:00
#SBATCH --constraint=L40S|A100
set -euo pipefail
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
python - <<'PY'
import traceback, torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
try:
    import pytorch3d
    print('pytorch3d top ok', pytorch3d.__file__)
except Exception:
    traceback.print_exc()
try:
    import pytorch3d.ops as ops
    print('pytorch3d.ops ok', ops.__file__)
except Exception:
    traceback.print_exc()
PY
