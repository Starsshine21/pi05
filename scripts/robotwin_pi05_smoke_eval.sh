#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task_name> <gpu_id> [task_config] [train_config_name] [model_name] [seed] [test_num]"
  exit 1
fi

task_name=${1}
gpu_id=${2}
task_config=${3:-demo_clean}
train_config_name=${4:-pi05_aloha_full_base}
model_name=${5:-model_robotwin}
seed=${6:-0}
test_num=${7:-1}

export CUDA_VISIBLE_DEVICES=${gpu_id}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp}
export ROBOTWIN_ASSET_ID=${ROBOTWIN_ASSET_ID:-robotwin}

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin
export PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/external/RoboTwin:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/src:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/packages/openpi-client/src:$PYTHONPATH

echo "[robotwin-pi05-smoke] task=${task_name} cfg=${task_config} train=${train_config_name} model=${model_name} ckpt=30000 asset=${ROBOTWIN_ASSET_ID}"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/pi05/deploy_policy.yml \
  --overrides \
  --task_name ${task_name} \
  --task_config ${task_config} \
  --train_config_name ${train_config_name} \
  --model_name ${model_name} \
  --ckpt_setting ${model_name} \
  --seed ${seed} \
  --policy_name pi05 \
  --test_num ${test_num}
