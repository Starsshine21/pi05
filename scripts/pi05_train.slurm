#!/bin/bash
#SBATCH -J pi05_train
#SBATCH -o /nfs_global/S/yangrongzheng/pi05/logs/pi05-train-%j.out
#SBATCH -e /nfs_global/S/yangrongzheng/pi05/logs/pi05-train-%j.err
#SBATCH -p r8nv-gpu
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00

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
source /nfs_global/S/yangrongzheng/pi05/scripts/use_local_openpi_env.sh

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
export PI05_RAY_PORT="${PI05_RAY_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
export PI05_RAY_DASHBOARD_PORT="${PI05_RAY_DASHBOARD_PORT:-$((40000 + ${SLURM_JOB_ID:-0} % 10000))}"
export PI05_RAY_CLIENT_PORT="${PI05_RAY_CLIENT_PORT:-$((45000 + ${SLURM_JOB_ID:-0} % 10000))}"
export RAY_ADDRESS="127.0.0.1:${PI05_RAY_PORT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false

export PI05_RESULTS_DIR="${PI05_RESULTS_DIR:-/nfs_global/S/yangrongzheng/pi05/results}"
export PI05_EXPERIMENT_NAME="${PI05_EXPERIMENT_NAME:-pi05_train_${SLURM_JOB_ID:-manual}}"
export PI05_MAX_STEPS="${PI05_MAX_STEPS:-20}"
export PI05_NUM_GROUP_ENVS="${PI05_NUM_GROUP_ENVS:-1}"
export PI05_EVAL_NUM_ENVS="${PI05_EVAL_NUM_ENVS:-1}"
export PI05_ROLLOUT_EPOCH="${PI05_ROLLOUT_EPOCH:-1}"
export PI05_UPDATE_EPOCH="${PI05_UPDATE_EPOCH:-1}"
export PI05_MICRO_BATCH_SIZE="${PI05_MICRO_BATCH_SIZE:-1}"
export PI05_GLOBAL_BATCH_SIZE="${PI05_GLOBAL_BATCH_SIZE:-1}"
export PI05_SAVE_INTERVAL="${PI05_SAVE_INTERVAL:-10}"
export PI05_SAFE_GET_LOGPROB="${PI05_SAFE_GET_LOGPROB:-true}"
export PI05_OPENPI_DTYPE="${PI05_OPENPI_DTYPE:-bfloat16}"

mkdir -p "${PI05_RESULTS_DIR}"

echo "PI05_RAY_PORT=${PI05_RAY_PORT}"
echo "PI05_RESULTS_DIR=${PI05_RESULTS_DIR}"
echo "PI05_EXPERIMENT_NAME=${PI05_EXPERIMENT_NAME}"
echo "PI05_MAX_STEPS=${PI05_MAX_STEPS}"
echo "PI05_NUM_GROUP_ENVS=${PI05_NUM_GROUP_ENVS}"
echo "PI05_SAFE_GET_LOGPROB=${PI05_SAFE_GET_LOGPROB}"
echo "PI05_OPENPI_DTYPE=${PI05_OPENPI_DTYPE}"

python -m ray.scripts.scripts stop --force || true
python -m ray.scripts.scripts start --head \
  --node-ip-address=127.0.0.1 \
  --port="${PI05_RAY_PORT}" \
  --dashboard-port="${PI05_RAY_DASHBOARD_PORT}" \
  --ray-client-server-port="${PI05_RAY_CLIENT_PORT}" \
  --num-cpus="${SLURM_CPUS_PER_TASK:-8}" \
  --num-gpus=1 \
  --disable-usage-stats \
  --include-dashboard=true \
  --temp-dir="${RAY_TMPDIR}"

cd ${PI05_REPO_ROOT}/local_vendor/RLinf-main/examples/embodiment
python train_embodied_agent.py \
  --config-name libero_10_ppo_openpi_pi05 \
  runner.max_steps="${PI05_MAX_STEPS}" \
  runner.save_interval="${PI05_SAVE_INTERVAL}" \
  runner.logger.log_path="${PI05_RESULTS_DIR}" \
  runner.logger.experiment_name="${PI05_EXPERIMENT_NAME}" \
  actor.checkpoint_save_path="${PI05_RESULTS_DIR}" \
  algorithm.num_group_envs="${PI05_NUM_GROUP_ENVS}" \
  env.eval.num_envs="${PI05_EVAL_NUM_ENVS}" \
  actor.micro_batch_size="${PI05_MICRO_BATCH_SIZE}" \
  actor.global_batch_size="${PI05_GLOBAL_BATCH_SIZE}" \
  algorithm.rollout_epoch="${PI05_ROLLOUT_EPOCH}" \
  algorithm.update_epoch="${PI05_UPDATE_EPOCH}" \
  rollout.pipeline_stage_num=1 \
  actor.model.openpi.safe_get_logprob="${PI05_SAFE_GET_LOGPROB}" \
  actor.model.openpi.dtype="${PI05_OPENPI_DTYPE}"

echo "Job end: $(date '+%F %T')"
