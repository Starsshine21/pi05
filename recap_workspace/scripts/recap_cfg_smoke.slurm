#!/bin/bash
#SBATCH -J recap_cfg_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap-cfg-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap-cfg-smoke-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --gres-flags=enforce-binding
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 1-00:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05/recap_workspace
PI05_ROOT=/nfs_global/S/yangrongzheng/pi05
VENV=/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/venv
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${RECAP_MASTER_PORT:-$((22000 + (${SLURM_JOB_ID:-0} % 20000)))}"
NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

mkdir -p "$ROOT/logs"

echo "RECAP cfg smoke start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES} WORLD_SIZE=${WORLD_SIZE}"

ulimit -l unlimited
ulimit -v unlimited

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh || true
source "$VENV/bin/activate"

export PYTHONPATH="$ROOT/external/rlinf-recap:${PYTHONPATH:-}"
export REPO_PATH="$ROOT/external/rlinf-recap"
export HF_HOME="${HF_HOME:-$PI05_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MPLCONFIGDIR="/tmp/mpl-rlinf-recap-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

echo "Runtime HF_HOME=${HF_HOME}"
echo "Runtime HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

srun --export=ALL --ntasks=${WORLD_SIZE} --ntasks-per-node=${GPUS_PER_NODE} bash -lc '
  source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh || true
  source /nfs_global/S/yangrongzheng/pi05/recap_workspace/external/venv/bin/activate
  export PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/recap_workspace:${PYTHONPATH:-}
  export REPO_PATH=/nfs_global/S/yangrongzheng/pi05/recap_workspace
  export HF_HOME=${HF_HOME}
  export HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
  export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
  export MPLCONFIGDIR=${MPLCONFIGDIR}
  export RANK=${SLURM_PROCID}
  export WORLD_SIZE='"${WORLD_SIZE}"'
  export LOCAL_RANK=${SLURM_LOCALID}
  export MASTER_ADDR='"${MASTER_ADDR}"'
  export MASTER_PORT='"${MASTER_PORT}"'
  echo "=== $(hostname) rank=${SLURM_PROCID} local_rank=${SLURM_LOCALID} ==="
  cd /nfs_global/S/yangrongzheng/pi05/recap_workspace
  cd /nfs_global/S/yangrongzheng/pi05/recap_workspace/examples/recap/cfg
  python train_cfg.py \
    --config-path /nfs_global/S/yangrongzheng/pi05/recap_workspace/configs \
    --config-name local_cfg_sft \
    runner.max_steps=1 \
    runner.max_epochs=1 \
    actor.micro_batch_size=1 \
    actor.global_batch_size=1 \
    data.num_workers=0
'

echo "RECAP cfg smoke end: $(date '+%F %T')"
