# PI05 LIBERO Simulation

This repo now has a local LIBERO simulation path for official `pi05_libero`.

## Data Links

- Pre-converted LeRobot dataset for training: https://huggingface.co/datasets/physical-intelligence/libero
- Raw LIBERO RLDS dataset, only needed if you want to reconvert data yourself: https://huggingface.co/datasets/openvla/modified_libero_rlds

The OpenPI config already points to `physical-intelligence/libero`, so the normal training path uses the pre-converted dataset.

Manual cache command:

```bash
cd /nfs_global/S/yangrongzheng/pi05
source scripts/use_pi05_libero_env.sh
huggingface-cli download physical-intelligence/libero --repo-type dataset
```

If you prefer a browser or another machine, download the same dataset from the HuggingFace page above and place it into the HuggingFace cache used by `HF_HOME=/nfs_global/S/yangrongzheng/pi05/.cache/huggingface`.

## Environment

```bash
cd /nfs_global/S/yangrongzheng/pi05
conda activate ./.conda-pi05-openpi-final
source scripts/use_pi05_libero_env.sh
python scripts/check_pi05_libero_env.py --env-smoke
```

The environment script sets:

- `PYTHONPATH` for `openpi_official`, `openpi-client`, and bundled `third_party/libero`
- `LIBERO_CONFIG_PATH` to `.cache/libero/config.yaml`
- `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`
- local writable cache dirs for HuggingFace, OpenPI, matplotlib, and numba

If EGL fails on a specific node, retry with:

```bash
MUJOCO_GL=glx PYOPENGL_PLATFORM=glx source scripts/use_pi05_libero_env.sh
```

## Training

After you have staged or allowed HuggingFace to cache `physical-intelligence/libero`, submit:

```bash
sbatch scripts/pi05_libero_train.slurm
```

Useful overrides:

```bash
sbatch --export=ALL,PI05_LIBERO_EXP_NAME=my_libero_run scripts/pi05_libero_train.slurm
sbatch --export=ALL,PI05_LIBERO_MAX_NORM_FRAMES=2000 scripts/pi05_libero_train.slurm
sbatch --export=ALL,PI05_LIBERO_COMPUTE_NORM_STATS=0 scripts/pi05_libero_train.slurm
```

Checkpoints are written under:

```text
results/openpi_libero_checkpoints/pi05_libero/<exp_name>/
```

## Deployment / Evaluation

Terminal 1, serve either the official checkpoint or your trained checkpoint:

```bash
source scripts/use_pi05_libero_env.sh
bash scripts/pi05_libero_serve_policy.sh

PI05_LIBERO_CHECKPOINT_DIR=/path/to/checkpoint/30000 \
  bash scripts/pi05_libero_serve_policy.sh
```

Terminal 2, run LIBERO evaluation:

```bash
source scripts/use_pi05_libero_env.sh
PI05_LIBERO_TASK_SUITE=libero_10 PI05_LIBERO_TRIALS=10 \
  bash scripts/pi05_libero_eval_client.sh
```

Videos are written to `results/libero/videos` by default.
