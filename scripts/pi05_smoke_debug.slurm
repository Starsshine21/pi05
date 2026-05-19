#!/bin/bash
#SBATCH -J pi05_dbg
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-dbg-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-dbg-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00

set -u

echo "STEP 1 start $(date '+%F %T')"
echo "STEP 2 host $(hostname)"
echo "STEP 3 cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
echo "STEP 4 sourced conda.sh"

conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
echo "STEP 5 activated conda"
echo "STEP 6 python $(which python)"
python -V

export EMBODIED_PATH=/nfs_global/S/yangrongzheng/RLinf-main/examples/embodiment
export REPO_PATH=/nfs_global/S/yangrongzheng/RLinf-main
export LIBERO_REPO_PATH=/nfs_global/S/yangrongzheng/RLinf-main/RLinf_deps/libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMNIGIBSON_HEADLESS=1
echo "STEP 7 env exported"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "STEP 8 nvidia-smi ok"

python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"
echo "STEP 9 torch ok"

python -c "import rlinf, openpi, lerobot, flax, jax; print('imports_ok')"
echo "STEP 10 imports ok"
