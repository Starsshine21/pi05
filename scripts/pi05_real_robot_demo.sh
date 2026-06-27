#!/usr/bin/env bash
set -euo pipefail

PI05_REPO_ROOT="${PI05_REPO_ROOT:-/nfs_global/S/yangrongzheng/pi05}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PI05_REPO_ROOT/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_pickplace_full_pytorch}"
PROMPT="${PROMPT:-pick and place}"
ROBOT_IP="${ROBOT_IP:-192.168.1.109}"
HAND_PORT="${HAND_PORT:-/dev/ttyUSB0}"
JOINT_DELTA_SCALE="${JOINT_DELTA_SCALE:-0.25}"
HAND_DELTA_SCALE="${HAND_DELTA_SCALE:-0.35}"
MAX_JOINT_DELTA="${MAX_JOINT_DELTA:-0.02}"
MAX_HAND_DELTA="${MAX_HAND_DELTA:-80}"
ACTION_EMA_ALPHA="${ACTION_EMA_ALPHA:-0.20}"
CONTROL_HZ="${CONTROL_HZ:-10}"
RECORD_DIR="${RECORD_DIR:-$PI05_REPO_ROOT/real_robot_demos}"
RECORD_FPS="${RECORD_FPS:-10}"
MAX_STEPS="${MAX_STEPS:-120}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate "$PI05_REPO_ROOT/.conda-pi05-openpi-final"
source "$PI05_REPO_ROOT/scripts/use_local_openpi_env.sh"

export PYTHONPATH="$PI05_REPO_ROOT/openpi_official/src:${PYTHONPATH:-}"
cd "$PI05_REPO_ROOT"

exec python "$PI05_REPO_ROOT/scripts/pi05_real_robot_infer.py" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --train-config "$TRAIN_CONFIG" \
  --prompt "$PROMPT" \
  --robot-ip "$ROBOT_IP" \
  --hand-port "$HAND_PORT" \
  --control-hz "$CONTROL_HZ" \
  --joint-delta-scale "$JOINT_DELTA_SCALE" \
  --hand-delta-scale "$HAND_DELTA_SCALE" \
  --max-joint-delta "$MAX_JOINT_DELTA" \
  --max-hand-delta "$MAX_HAND_DELTA" \
  --action-ema-alpha "$ACTION_EMA_ALPHA" \
  --record-dir "$RECORD_DIR" \
  --record-fps "$RECORD_FPS" \
  --max-steps "$MAX_STEPS"
