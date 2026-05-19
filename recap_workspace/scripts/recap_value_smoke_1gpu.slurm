#!/bin/bash
#SBATCH -J recap_value_smoke1
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap-value-smoke1-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap-value-smoke1-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 1-00:00:00

set -euo pipefail

ROOT=/nfs_global/S/yangrongzheng/pi05/recap_workspace
PI05_ROOT=/nfs_global/S/yangrongzheng/pi05
VENV=/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/venv
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${RECAP_MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}"
RAY_PORT="${RECAP_RAY_PORT:-$((10000 + (${SLURM_JOB_ID:-0} % 20000)))}"

mkdir -p "$ROOT/logs"

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

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_LEROBOT_HOME" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$RECAP_JOB_LOG_DIR"

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
  runner.max_steps=1 \
  runner.max_epochs=1 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=1 \
  data.train_num_workers=0 \
  data.eval_num_workers=0

ray stop --force >/dev/null 2>&1 || true
