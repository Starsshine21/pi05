#!/bin/bash
#SBATCH -J pistar06_infer_pickplace
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pistar06-infer-pickplace-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pistar06-infer-pickplace-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00

set -euo pipefail

export PI05_REPO_ROOT=/nfs_global/S/yangrongzheng/pi05
export EVORL_ROOT=/nfs_global/S/yangrongzheng/evo-RL/Evo-RL
export LEROBOT_DATA_DIR="${PI05_REPO_ROOT}/data/lerobot_pick_place_v30"
export HF_HOME="${HF_HOME:-${PI05_REPO_ROOT}/.cache/huggingface}"

export CHECKPOINT_PATH="${PI05_REPO_ROOT}/results/pistar06_pickplace_20260626_131915/checkpoints/008000"
export OUTPUT_DIR="${PI05_REPO_ROOT}/results/pistar06_pickplace_infer"
export ACP_N_STEP="${ACP_N_STEP:-50}"
export ACP_POSITIVE_RATIO="${ACP_POSITIVE_RATIO:-0.3}"
export INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"

mkdir -p "${PI05_REPO_ROOT}/logs" "${OUTPUT_DIR}"

echo "=== Pistar06 Value Inference ==="
echo "Start: $(date '+%F %T')"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data: ${LEROBOT_DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "ACP n_step: ${ACP_N_STEP}"
echo "ACP positive_ratio: ${ACP_POSITIVE_RATIO}"

ulimit -l unlimited || true
ulimit -v unlimited || true

source /nfs_global/S/yangrongzheng/miniconda3-fresh/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/evo-RL/dexjoco/.conda/openpi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${EVORL_ROOT}/src:${PI05_REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

cd "${EVORL_ROOT}"

python -m lerobot.scripts.lerobot_value_infer \
  --dataset.repo_id=local/lerobot_pick_place_v30 \
  --dataset.root="${LEROBOT_DATA_DIR}" \
  --dataset.success_field=episode_success \
  --dataset.default_success=success \
  --inference.checkpoint_path="${CHECKPOINT_PATH}" \
  --runtime.device=cuda \
  --runtime.batch_size="${INFER_BATCH_SIZE}" \
  --runtime.num_workers=0 \
  --acp.enable=true \
  --acp.n_step="${ACP_N_STEP}" \
  --acp.positive_ratio="${ACP_POSITIVE_RATIO}" \
  --acp.value_field=complementary_info.value \
  --acp.advantage_field=complementary_info.advantage \
  --acp.indicator_field=complementary_info.acp_indicator \
  --acp.c_fail_coef=1.0 \
  --viz.enable=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=pistar06_infer_pickplace

echo "=== Done ==="
echo "End: $(date '+%F %T')"
