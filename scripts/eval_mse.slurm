#!/bin/bash
#SBATCH -J pi05_eval_mse
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-eval-mse-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-eval-mse-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00

set -euo pipefail

export PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
export DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place"
export CHECKPOINT_BASE="${PI05_REPO_ROOT}/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch"

mkdir -p "${PI05_REPO_ROOT}/logs"

ulimit -l unlimited
ulimit -v unlimited

CONDA_SH="$(find /home/S/yangrongzheng/miniconda3* /nfs_global/S/yangrongzheng/miniconda3* -maxdepth 3 -name 'conda.sh' -path '*/etc/profile.d/*' 2>/dev/null | head -1)"
if [[ -z "${CONDA_SH}" ]]; then
  CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
source "${CONDA_SH}"
conda activate "${PI05_REPO_ROOT}/.conda-pi05-openpi-final"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EVAL_SCRIPT="${PI05_REPO_ROOT}/scripts/eval_mse.py"
MAX_EPISODES="${MAX_EPISODES:-50}"
NUM_STEPS="${NUM_STEPS:-10}"

CHECKPOINTS=(
  "${CHECKPOINT_BASE}/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/10000"
  "${CHECKPOINT_BASE}/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/40000"
  "${CHECKPOINT_BASE}/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/50000"
  "${CHECKPOINT_BASE}/pi05_pickplace_full_pytorch_eefdelta_v2_freshnorm_train/60000"
)

echo "PI0.5 eval MSE start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "PI05 env: $(conda info --envs 2>/dev/null | grep 'pi05.*final')"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'N/A')"
echo "MAX_EPISODES=${MAX_EPISODES}"
echo "NUM_STEPS=${NUM_STEPS}"

# Fix torch for cluster GPU driver
echo "=== Checking torch CUDA compatibility ==="
python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null || {
  echo "torch CUDA not available, reinstalling..."
  pip3 uninstall torch -y 2>/dev/null
  pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
  echo "torch reinstalled: $(python3 -c 'import torch; print(torch.__version__)')"
}

# Ensure transformers_replace patch is applied
TRANSFORMERS_DIR="$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')"
if [[ ! -f "${TRANSFORMERS_DIR}/models/gemma/modeling_gemma.py.patched" ]]; then
  echo "Applying transformers_replace patch..."
  cp -r "${PI05_REPO_ROOT}/openpi_official/src/openpi/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/"
  touch "${TRANSFORMERS_DIR}/models/gemma/modeling_gemma.py.patched"
fi

echo "CHECKPOINTS:"
printf '  %s\n' "${CHECKPOINTS[@]}"
echo "===================="

python3 "${EVAL_SCRIPT}" \
  --ckpt-dirs "${CHECKPOINTS[@]}" \
  --data-dir "${DATA_DIR}" \
  --max-episodes "${MAX_EPISODES}" \
  --num-steps "${NUM_STEPS}" \
  --device cuda

echo "PI0.5 eval MSE end: $(date '+%F %T')"
