#!/bin/bash
#SBATCH -J pi05_smoke
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-smoke-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-smoke-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00

set -e

mkdir -p /nfs_global/S/yangrongzheng/pi05/logs

echo "STEP 1 start $(date '+%F %T')"
echo "STEP 2 host $(hostname)"
echo "STEP 3 cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export EMBODIED_PATH=${PI05_REPO_ROOT}/local_vendor/RLinf-main/examples/embodiment
export REPO_PATH=${PI05_REPO_ROOT}/local_vendor/RLinf-main
export LIBERO_REPO_PATH=${PI05_REPO_ROOT}/local_vendor/RLinf-main/RLinf_deps/libero
export PYTHONPATH="${LIBERO_REPO_PATH}:${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMNIGIBSON_HEADLESS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "STEP 4 gpu ok"

python - <<'PY'
import torch
from omegaconf import OmegaConf
from rlinf.models import get_model

print("torch_version", torch.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device_count", torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0), flush=True)

cfg = OmegaConf.create(
    {
        "model_name": "openpi",
        "precision": None,
        "num_action_chunks": 10,
        "action_dim": 12,
        "add_value_head": True,
        "is_lora": False,
        "lora_rank": 32,
        "gradient_checkpointing": False,
        "use_wrist_image": True,
        "use_proprio": True,
        "openpi": {
            "simulator_type": "libero",
            "pi05": True,
            "noise_level": 0.5,
            "action_chunk": 10,
            "num_steps": 5,
            "train_expert_only": True,
            "action_env_dim": 12,
            "noise_method": "flow_sde",
            "add_value_head": True,
            "value_after_vlm": True,
            "value_vlm_mode": "mean_token",
        },
    }
)

model = get_model("/nfs_global/S/yangrongzheng/pi05/pi05_sft", cfg)
print(type(model).__name__, flush=True)
print("loaded_ok", flush=True)
PY

echo "STEP 5 model loaded"
echo "Job end: $(date '+%F %T')"
