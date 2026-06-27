#!/bin/bash
#SBATCH -J pi05_train_diag
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-train-diag-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-train-diag-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00

set -euo pipefail

mkdir -p /nfs_global/S/yangrongzheng/pi05/logs

echo "Job start: $(date '+%F %T')"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_TIMELIMIT=${SLURM_TIMELIMIT:-}"

ulimit -l unlimited
ulimit -v unlimited

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

cleanup() {
  python -m ray.scripts.scripts stop --force || true
}
trap cleanup EXIT

export EMBODIED_PATH=${PI05_REPO_ROOT}/local_vendor/RLinf-main/examples/embodiment
export REPO_PATH=${PI05_REPO_ROOT}/local_vendor/RLinf-main
export LIBERO_REPO_PATH=${PI05_REPO_ROOT}/local_vendor/RLinf-main/RLinf_deps/libero
export PYTHONPATH="${LIBERO_REPO_PATH}:${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMNIGIBSON_HEADLESS=1
export RAY_DEDUP_LOGS=0
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export RAY_TMPDIR=/tmp/ray-${SLURM_JOB_ID}
export TMPDIR=/tmp
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset ALL_PROXY_HTTP ALL_PROXY_HTTPS all_proxy_http all_proxy_https
export RAY_ADDRESS="127.0.0.1:6379"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python -m ray.scripts.scripts stop --force || true
python -m ray.scripts.scripts start --head \
  --node-ip-address=127.0.0.1 \
  --port=6379 \
  --num-cpus="${SLURM_CPUS_PER_TASK:-8}" \
  --num-gpus=1 \
  --disable-usage-stats \
  --include-dashboard=true \
  --temp-dir="${RAY_TMPDIR}"

cd ${PI05_REPO_ROOT}/local_vendor/RLinf-main/examples/embodiment
python train_embodied_agent.py \
  --config-name libero_10_ppo_openpi_pi05 \
  runner.max_steps=2 \
  algorithm.num_group_envs=1 \
  env.eval.num_envs=1 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=1 \
  algorithm.rollout_epoch=1 \
  algorithm.update_epoch=1

echo "Job end: $(date '+%F %T')"
