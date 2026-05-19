# PI05 训练启动说明

这份文档面向**当前仓库这条已经打通的 pi05 复现训练链路**，目标是让你从零开始，按现在仓库里的实际脚本把训练跑起来。

当前训练主入口是：

- `scripts/pi05_train.slurm`：正式训练入口
- `scripts/pi05_train_smoke.slurm`：最小 smoke test
- `scripts/pi05_train_diag.slurm`：短步数诊断入口
- `scripts/pi05_smoke.slurm`：环境 + 模型加载检查

训练实际调用的是 RLinf 里的：

- `RLinf-main/examples/embodiment/train_embodied_agent.py`
- 配置名：`libero_10_ppo_openpi_pi05`

---

## 1. 目录和依赖关系

这套复现不是一个完全独立的训练框架，而是**当前仓库 + 本地 RLinf 环境**拼起来跑：

- 当前仓库根目录：`/nfs_global/S/yangrongzheng/pi05`
- RLinf 仓库：`/nfs_global/S/yangrongzheng/RLinf-main`
- conda 环境：`/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final`
- PI05 SFT 模型目录：`/nfs_global/S/yangrongzheng/pi05/pi05_sft`

脚本里已经把这些路径基本写死了，所以最稳妥的方式是按当前目录结构直接使用。

---

## 2. 开始前先确认什么

正式提交训练前，建议至少确认下面四件事：

### 2.1 conda 环境存在

检查：

```bash
ls /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final/bin/python
```

如果环境还没建，可以重建：

```bash
bash scripts/create_conda_env.sh
```

这个脚本会创建：

- `/.conda-pi05-openpi-final`

并把 RLinf 的 site-packages 通过 `.pth` 方式桥接进来。

### 2.2 RLinf 代码和虚拟环境存在

至少要确认这些目录存在：

```bash
ls /nfs_global/S/yangrongzheng/RLinf-main
ls /nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages
ls /nfs_global/S/yangrongzheng/RLinf-main/examples/embodiment
```

### 2.3 pi05 权重目录存在

检查：

```bash
ls /nfs_global/S/yangrongzheng/pi05/pi05_sft
```

`scripts/pi05_smoke.slurm` 里会直接从这个目录加载模型。

### 2.4 本地环境变量拼接正常

运行：

```bash
bash scripts/check_local_openpi_env.sh
```

它会：

- source `scripts/use_local_openpi_env.sh`
- 检查 `OPENPI_LOCAL_PYTHON`
- 检查 RLinf site-packages 是否可见
- 检查本地 Python 环境是否能正确拿到依赖

如果这一步不过，不建议直接提交训练。

---

## 3. 推荐启动顺序

建议按下面顺序来，不要一上来就跑正式训练。

### 第一步：做基础环境 smoke

```bash
sbatch scripts/pi05_smoke.slurm
```

这个脚本主要检查：

- GPU 是否可见
- `torch.cuda.is_available()` 是否正常
- `get_model(...)` 能不能从 `pi05_sft` 正常加载 openpi/pi05 模型

如果这个不过，先别跑训练。

### 第二步：做训练链路诊断

```bash
sbatch scripts/pi05_train_diag.slurm
```

这个脚本会：

- 启动本地 Ray head
- 进入 `RLinf-main/examples/embodiment`
- 用 `libero_10_ppo_openpi_pi05` 跑一个很短的 2 step 训练

适合排查：

- Ray 起不起来
- 配置能不能被正确加载
- 训练主循环能不能走通

### 第三步：做正式 smoke train

```bash
sbatch scripts/pi05_train_smoke.slurm
```

这个脚本是更接近正式训练配置的最小训练版本，通常用于确认：

- 训练参数覆盖项是否合理
- 日志路径/结果路径是否正常写入
- 当前模型精度设置是否可用

### 第四步：提交正式训练

```bash
sbatch scripts/pi05_train.slurm
```

这是当前仓库里的正式训练入口。

---

## 4. 正式训练到底做了什么

`scripts/pi05_train.slurm` 的主要流程如下：

1. 创建日志目录 `logs/`
2. 激活 conda：`/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final`
3. source `scripts/use_local_openpi_env.sh`
4. 设置 RLinf / LIBERO / MuJoCo / EGL / Ray 相关环境变量
5. 在本机启动一个独立 Ray head
6. 进入 `RLinf-main/examples/embodiment`
7. 执行：

```bash
python train_embodied_agent.py \
  --config-name libero_10_ppo_openpi_pi05
```

并通过命令行覆盖一批关键训练参数，例如：

- `runner.max_steps`
- `runner.logger.log_path`
- `runner.logger.experiment_name`
- `actor.checkpoint_save_path`
- `algorithm.num_group_envs`
- `env.eval.num_envs`
- `actor.micro_batch_size`
- `actor.global_batch_size`
- `algorithm.rollout_epoch`
- `algorithm.update_epoch`
- `rollout.pipeline_stage_num`
- `actor.model.openpi.safe_get_logprob`
- `actor.model.openpi.dtype`

也就是说，**真正的默认配置在 RLinf 的 hydra config 里，而这个 slurm 脚本负责覆盖当前复现所需的最关键参数**。

---

## 5. 最常用的启动命令

### 5.1 提交正式训练

```bash
sbatch scripts/pi05_train.slurm
```

### 5.2 自定义实验名

```bash
sbatch --export=ALL,PI05_EXPERIMENT_NAME=my_pi05_run scripts/pi05_train.slurm
```

### 5.3 指定模型精度

默认脚本里：

- `PI05_OPENPI_DTYPE` 默认是 `bfloat16`

如果想改成 `float32`：

```bash
sbatch --export=ALL,PI05_OPENPI_DTYPE=float32 scripts/pi05_train.slurm
```

### 5.4 改训练步数做短测

```bash
sbatch --export=ALL,PI05_MAX_STEPS=2,PI05_EXPERIMENT_NAME=pi05_train_2step_diag scripts/pi05_train.slurm
```

> 注意：是否生效取决于脚本里有没有读取该环境变量。当前仓库里更稳妥的做法，还是直接使用现成的 `scripts/pi05_train_diag.slurm` 或 `scripts/pi05_train_smoke.slurm`。

---

## 6. 日志和结果在哪里看

### 日志目录

slurm 日志默认在：

```bash
/nfs_global/S/yangrongzheng/pi05/logs
```

常见文件：

- `logs/pi05-train-<jobid>.out`
- `logs/pi05-train-<jobid>.err`
- `logs/pi05-train-diag-<jobid>.out`
- `logs/pi05-train-diag-<jobid>.err`
- `logs/pi05-smoke-<jobid>.out`
- `logs/pi05-smoke-<jobid>.err`

查看方式：

```bash
tail -f logs/pi05-train-<jobid>.out
```

```bash
tail -f logs/pi05-train-<jobid>.err
```

### 结果目录

`scripts/pi05_train.slurm` 中默认：

```bash
PI05_RESULTS_DIR=/nfs_global/S/yangrongzheng/pi05/results
```

实验名默认：

```bash
PI05_EXPERIMENT_NAME=pi05_smoke_<jobid>
```

所以结果通常会落在：

```bash
/nfs_global/S/yangrongzheng/pi05/results
```

如果你传了新的 `PI05_EXPERIMENT_NAME`，就按你的实验名区分。

---

## 7. 训练前后最实用的排查命令

### 查看队列

```bash
squeue -u $USER
```

### 查看某个任务

```bash
squeue -j <jobid>
```

### 实时看日志

```bash
tail -f logs/pi05-train-<jobid>.out
```

```bash
tail -f logs/pi05-train-<jobid>.err
```

### 看 GPU

```bash
nvidia-smi
```

### 手动检查环境是否能起

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
source scripts/use_local_openpi_env.sh
bash scripts/check_local_openpi_env.sh
```

---

## 8. 常见问题

### 8.1 `pi05_smoke.slurm` 能过，但训练起不来

通常优先检查：

- Ray 端口是否冲突
- `RAY_TMPDIR=/tmp/ray-${SLURM_JOB_ID}` 是否可写
- RLinf 的配置 `libero_10_ppo_openpi_pi05` 是否还在
- `examples/embodiment/train_embodied_agent.py` 是否被改动过

### 8.2 报找不到 RLinf 包

优先检查：

- `scripts/use_local_openpi_env.sh` 是否成功执行
- `RLINF_ROOT` 是否还是 `/nfs_global/S/yangrongzheng/RLinf-main`
- `RLINF_VENV_SITE_PACKAGES` 是否存在
- `PYTHONPATH` 是否包含 RLinf 和 libero 路径

### 8.3 报模型权重找不到

优先检查：

- `pi05_sft/` 目录是否存在
- 权重是不是完整下载
- 加载代码是否仍然读取 `/nfs_global/S/yangrongzheng/pi05/pi05_sft`

### 8.4 想先验证不是数据/配置问题，而是环境问题

最推荐顺序：

1. `bash scripts/check_local_openpi_env.sh`
2. `sbatch scripts/pi05_smoke.slurm`
3. `sbatch scripts/pi05_train_diag.slurm`
4. `sbatch scripts/pi05_train.slurm`

---

## 9. 一套最简开始训练流程

如果你现在已经确认这套复现“能跑”，那最简流程就是：

```bash
cd /nfs_global/S/yangrongzheng/pi05
bash scripts/check_local_openpi_env.sh
sbatch scripts/pi05_smoke.slurm
sbatch scripts/pi05_train_diag.slurm
sbatch --export=ALL,PI05_EXPERIMENT_NAME=pi05_run_001 scripts/pi05_train.slurm
```

提交后再看：

```bash
squeue -u $USER
tail -f logs/pi05-train-<jobid>.out
tail -f logs/pi05-train-<jobid>.err
```

---

## 10. 这份文档对应的仓库事实

这份说明基于当前仓库里的真实入口整理，核心参考文件如下：

- `README.md`
- `scripts/pi05_train.slurm`
- `scripts/pi05_train_smoke.slurm`
- `scripts/pi05_train_diag.slurm`
- `scripts/pi05_smoke.slurm`
- `scripts/use_local_openpi_env.sh`
- `scripts/check_local_openpi_env.sh`
- `scripts/create_conda_env.sh`

如果后面你改了 RLinf 路径、conda 环境名、结果目录或者 slurm 参数，这份文档也要一起更新。
