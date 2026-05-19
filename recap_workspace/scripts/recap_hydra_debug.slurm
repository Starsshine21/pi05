#!/bin/bash
#SBATCH -J recap_hydra_dbg
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap-hydra-dbg-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/recap_workspace/logs/recap-hydra-dbg-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --nodes=1
#SBATCH --qos=gpu-short
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-00:30:00
set -euo pipefail
ROOT=/nfs_global/S/yangrongzheng/pi05/recap_workspace
PI05_ROOT=/nfs_global/S/yangrongzheng/pi05
VENV=/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/venv
mkdir -p "$ROOT/logs"
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh || true
source "$VENV/bin/activate"
export PYTHONPATH="$ROOT/external/rlinf-recap:${PYTHONPATH:-}"
export REPO_PATH="$ROOT/external/rlinf-recap"
export HF_HOME="${HF_HOME:-$PI05_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PI05_ROOT/.cache/huggingface/lerobot}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MPLCONFIGDIR="/tmp/mpl-rlinf-recap-hydra-${SLURM_JOB_ID:-manual}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_LEROBOT_HOME" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR"
cd "$PI05_ROOT"
python /nfs_global/S/yangrongzheng/pi05/recap_workspace/hydra_probe.py
