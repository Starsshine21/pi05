#!/usr/bin/env bash
set -euo pipefail

PI05_REPO_ROOT="${PI05_REPO_ROOT:-/nfs_global/S/yangrongzheng/pi05}"
OPENPI_ROOT="${OPENPI_ROOT:-$PI05_REPO_ROOT/openpi_official}"
LEROBOT_DATA_DIR="${LEROBOT_DATA_DIR:-$PI05_REPO_ROOT/data/lerobot_pick_place}"
HF_LEROBOT_ROOT="${HF_LEROBOT_ROOT:-/nfs_global/S/yangrongzheng/RLinf-main/models/huggingface/lerobot}"
LEROBOT_LINK="${LEROBOT_LINK:-$HF_LEROBOT_ROOT/local/pi05-pickplace-il}"
CONFIG_NAME="${CONFIG_NAME:-pi05_pickplace_lora_pytorch}"
EXP_NAME="${EXP_NAME:-pickplace_pi05_lora_pytorch}"
MAX_NORM_FRAMES="${MAX_NORM_FRAMES:-}"
RUN_NORM_STATS="${RUN_NORM_STATS:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LOG_INTERVAL="${LOG_INTERVAL:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-}"
OVERWRITE="${OVERWRITE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
TORCHRUN_NPROC_PER_NODE="${TORCHRUN_NPROC_PER_NODE:-}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "$PI05_REPO_ROOT/.conda-pi05-openpi-final"
source "$PI05_REPO_ROOT/scripts/use_local_openpi_env.sh"

export PYTHONPATH="$OPENPI_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/mpl-openpi
mkdir -p "$MPLCONFIGDIR"

if [[ ! -d "$LEROBOT_DATA_DIR/meta" ]]; then
  echo "Missing LeRobot dataset at $LEROBOT_DATA_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$LEROBOT_LINK")"
if [[ -L "$LEROBOT_LINK" ]]; then
  current_target="$(readlink "$LEROBOT_LINK")"
  if [[ "$current_target" != "$LEROBOT_DATA_DIR" ]]; then
    rm -f "$LEROBOT_LINK"
    ln -s "$LEROBOT_DATA_DIR" "$LEROBOT_LINK"
  fi
elif [[ -e "$LEROBOT_LINK" ]]; then
  echo "$LEROBOT_LINK exists and is not a symlink; refusing to replace it." >&2
  exit 1
else
  ln -s "$LEROBOT_DATA_DIR" "$LEROBOT_LINK"
fi

cd "$OPENPI_ROOT"

NORM_STATS_BASE="${ASSETS_BASE_DIR:-assets}"
NORM_STATS_PATH="$NORM_STATS_BASE/pi05_pickplace_lora/local/pi05-pickplace-il/norm_stats.json"
if [[ "$RUN_NORM_STATS" == "1" && ! -f "$NORM_STATS_PATH" ]]; then
  norm_cmd=(python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME")
  if [[ -n "$MAX_NORM_FRAMES" ]]; then
    norm_cmd+=(--max-frames "$MAX_NORM_FRAMES")
  fi
  if [[ -n "$NUM_WORKERS" ]]; then
    norm_cmd+=(--num-workers "$NUM_WORKERS")
  fi
  if [[ -n "$ASSETS_BASE_DIR" ]]; then
    norm_cmd+=(--assets-base-dir "$ASSETS_BASE_DIR")
  fi
  echo "[pi05-pytorch] Running norm stats: ${norm_cmd[*]}"
  "${norm_cmd[@]}"
elif [[ -f "$NORM_STATS_PATH" ]]; then
  echo "[pi05-pytorch] Reusing norm stats: $NORM_STATS_PATH"
else
  echo "[pi05-pytorch] Skipping norm stats because RUN_NORM_STATS=$RUN_NORM_STATS"
fi

if [[ "$RUN_TRAIN" != "1" ]]; then
  echo "[pi05-pytorch] Skipping training because RUN_TRAIN=$RUN_TRAIN"
  exit 0
fi

train_cmd=(python scripts/train_pytorch.py "$CONFIG_NAME" --exp_name "$EXP_NAME")
if [[ "$WANDB_ENABLED" == "0" ]]; then
  train_cmd+=(--no-wandb-enabled)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  train_cmd+=(--overwrite)
fi
if [[ -n "$NUM_TRAIN_STEPS" ]]; then
  train_cmd+=(--num_train_steps "$NUM_TRAIN_STEPS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  train_cmd+=(--batch_size "$BATCH_SIZE")
fi
if [[ -n "$LOG_INTERVAL" ]]; then
  train_cmd+=(--log_interval "$LOG_INTERVAL")
fi
if [[ -n "$SAVE_INTERVAL" ]]; then
  train_cmd+=(--save_interval "$SAVE_INTERVAL")
fi
if [[ -n "$NUM_WORKERS" ]]; then
  train_cmd+=(--num_workers "$NUM_WORKERS")
fi

if [[ -n "$TORCHRUN_NPROC_PER_NODE" ]]; then
  launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="$TORCHRUN_NPROC_PER_NODE")
else
  launcher=()
fi

echo "[pi05-pytorch] Running train: ${launcher[*]} ${train_cmd[*]}"
"${launcher[@]}" "${train_cmd[@]}"
