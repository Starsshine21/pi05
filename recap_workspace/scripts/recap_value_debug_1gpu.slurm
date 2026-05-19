#!/bin/bash
#SBATCH -J recap_value_dbg1
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap-value-dbg1-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap-value-dbg1-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --nodes=1
#SBATCH --qos=gpu-short
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 0-04:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05/recap_workspace
PI05_ROOT=/nfs_global/S/yangrongzheng/pi05
VENV=/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/venv
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${RECAP_MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"
RAY_PORT="${RECAP_RAY_PORT:-$((10000 + (${SLURM_JOB_ID:-0} % 20000)))}"

mkdir -p "$ROOT/logs"

echo "RECAP value 1gpu debug start: $(date +'%F %T')"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RAY_PORT=${RAY_PORT}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh || true
source "$VENV/bin/activate"

export PYTHONPATH="$ROOT/external/rlinf-recap:${PYTHONPATH:-}"
export REPO_PATH="$ROOT/external/rlinf-recap"
export HF_HOME="${HF_HOME:-$PI05_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PI05_ROOT/.cache/huggingface/lerobot}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MPLCONFIGDIR="/tmp/mpl-rlinf-recap-${SLURM_JOB_ID:-manual}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES="0"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RECAP_JOB_LOG_DIR="$ROOT/logs/recap_value_${SLURM_JOB_ID:-manual}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

ulimit -n 8192 || true

echo "Runtime HF_HOME=${HF_HOME}"
echo "Runtime HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "Runtime HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "Runtime CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_LEROBOT_HOME" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$RECAP_JOB_LOG_DIR"

echo "Runtime RECAP_JOB_LOG_DIR=${RECAP_JOB_LOG_DIR}"

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address="${MASTER_ADDR}" --port="${RAY_PORT}" --num-cpus=8 --num-gpus=1 --memory=200000000000 --object-store-memory=4000000000 --temp-dir="/tmp/ray-${SLURM_JOB_ID:-manual}" --include-dashboard=true --include-log-monitor=false --disable-usage-stats
export RAY_ADDRESS="${MASTER_ADDR}:${RAY_PORT}"

cd "$PI05_ROOT"
python /nfs_global/S/yangrongzheng/pi05/recap_workspace/run_value_local.py \
  --config-path /nfs_global/S/yangrongzheng/pi05/recap_workspace/configs \
  --config-name local_value_sft \
  cluster.num_nodes=1 \
  +runner.per_worker_log=true \
  runner.logger.log_path=${RECAP_JOB_LOG_DIR} \
  actor.micro_batch_size=1 \
  actor.global_batch_size=1 \
  actor.model.precision=fp32 \
  data.train_num_workers=0 \
  data.eval_num_workers=0 \
  runner.max_steps=1 \
  runner.max_epochs=1

ray stop --force >/dev/null 2>&1 || true

echo "RECAP value 1gpu debug end: $(date +'%F %T')"
