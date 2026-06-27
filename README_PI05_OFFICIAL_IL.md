# PI05 官方 IL 训练进展与使用说明

这份文档记录当前仓库中 **基于官方 `openpi` 的 `pi05` VLA imitation learning** 进展、目录结构、运行方法与排查方式。

## 1. 当前项目进度

目前已经完成：

- 原始 `pkl` 数据转为本地 LeRobot 数据集
- 本地 LeRobot 数据接到官方 `openpi` 训练链路
- 官方 `pi05` 自定义训练配置注册完成
- 官方 `compute_norm_stats` 路线已跑通到数据读取阶段
- 官方 `train.py` 已成功进入训练初始化阶段

当前目标路线是：

- **只以官方 `openpi/pi05` 为准**
- 使用 **文本 + 图像 + state -> action** 的 VLA imitation learning

不是之前的 state-only baseline。

---

## 2. 关键目录

### 2.1 当前仓库根目录

```bash
/nfs_global/S/yangrongzheng/pi05
```

### 2.2 转换后的 LeRobot 数据

```bash
/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place
```

### 2.3 官方 openpi 仓库

```bash
/nfs_global/S/yangrongzheng/pi05/openpi_official
```

### 2.4 PI05 权重目录

```bash
/nfs_global/S/yangrongzheng/pi05/pi05_sft
```

---

## 3. 现在官方 IL 的输入输出是什么

当前已经对齐到官方 `pi05` VLA 输入范式。

### 输入

- 文本：来自 LeRobot 数据中的 `task`
- 第三人称图像：`observation.images.l515`
- wrist 图像：`observation.images.orbbec`
- 状态：`observation.state`

通过官方 `openpi` 配置映射后，进入模型的格式为：

- `prompt`
- `observation/image`
- `observation/wrist_image`
- `observation/state`

### 输出

- `action`
- 当前 action 维度是 `7`

---

## 4. 当前已经接好的官方配置

官方自定义配置名：

```bash
pi05_pickplace_lora
```

相关文件：

- `openpi_official/src/openpi/training/config.py`
- `openpi_official/src/openpi/training/pi05_pickplace_config.py`

这份配置做了下面几件事：

- 使用官方 `pi05` LoRA 模型配置
- 使用你本地的 LeRobot 数据：`local/pi05-pickplace-il`
- 把你的数据字段重映射到官方 `libero/pi05` 风格输入
- 关闭 full finetune，走 LoRA 路线

---

## 5. 从头复现的完整流程

### 第一步：切环境

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
source /nfs_global/S/yangrongzheng/pi05/scripts/use_local_openpi_env.sh
```

### 第二步：如果需要，重新转换数据

```bash
python scripts/convert_pick_place_to_lerobot.py --overwrite --image-height 224 --image-width 224
```

### 第三步：确保官方 loader 能找到本地 LeRobot 数据

```bash
mkdir -p /nfs_global/S/yangrongzheng/RLinf-main/models/huggingface/lerobot/local
rm -rf /nfs_global/S/yangrongzheng/RLinf-main/models/huggingface/lerobot/local/pi05-pickplace-il
ln -s /nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place /nfs_global/S/yangrongzheng/RLinf-main/models/huggingface/lerobot/local/pi05-pickplace-il
```

### 第四步：进入官方 openpi 仓库

```bash
cd /nfs_global/S/yangrongzheng/pi05/openpi_official
```

### 第五步：计算 norm stats

```bash
PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/openpi_official/src:$PYTHONPATH \
python scripts/compute_norm_stats.py --config-name pi05_pickplace_lora
```

### 第六步：启动官方 `pi05` LoRA 训练

```bash
PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/openpi_official/src:$PYTHONPATH \
python scripts/train.py pi05_pickplace_lora --exp-name pickplace_pi05_lora --no-wandb-enabled --overwrite
```

也可以直接使用已经写好的脚本：

```bash
./run_pi05_pickplace_lora.sh
```

如果只想先验证数据链路和 norm stats，不进入完整训练：

```bash
./scripts/smoke_pi05_official_il.sh
```

默认 smoke 会设置：

- `MAX_NORM_FRAMES=1024`
- `NUM_WORKERS=0`
- `ASSETS_BASE_DIR=assets_smoke`
- `RUN_TRAIN=0`

在设置 `MAX_NORM_FRAMES` 时，当前 `compute_norm_stats.py` 会只打开覆盖这些 frame 所需的前几个 episode，不再为了 smoke 先扫完整 26GB 数据集。
smoke 的统计文件会写到 `openpi_official/assets_smoke/`，不会污染完整训练使用的 `openpi_official/assets/`。

如果要做 2 step 训练 smoke：

```bash
RUN_TRAIN=1 NUM_TRAIN_STEPS=2 LOG_INTERVAL=1 SAVE_INTERVAL=1 EXP_NAME=pickplace_pi05_lora_2step ./scripts/smoke_pi05_official_il.sh
```

完整复现入口也可以从仓库根目录运行：

```bash
./scripts/reproduce_pi05_official_il.sh
```

---

## 6. 训练日志怎么看

当前官方训练不是 slurm 版脚本，而是前台 Python 进程，所以主要通过下面方式查看：

### 第一次启动为什么会很久

第一次读取本地 LeRobot 数据时，`datasets` 会把 496 个 parquet episode 构建成本地缓存。终端会出现类似：

```text
Generating train split: 175123 examples
```

这一步是数据缓存构建，不是模型训练。缓存写到：

```bash
/nfs_global/S/yangrongzheng/pi05/.cache/huggingface/datasets
```

后续再启动通常会快很多。

### norm stats 进度

`compute_norm_stats.py` 会显示 tqdm：

```text
Computing stats:  12%|...
progress: batch 100/...
```

也可以用小样本先验证：

```bash
MAX_NORM_FRAMES=1024 RUN_TRAIN=0 ./scripts/reproduce_pi05_official_il.sh
```

### train 进度

训练真正开始后会按 `log_interval` 输出 step/loss。短 smoke 可以这样跑：

```bash
RUN_TRAIN=1 NUM_TRAIN_STEPS=2 LOG_INTERVAL=1 SAVE_INTERVAL=1 ./scripts/smoke_pi05_official_il.sh
```

### 查看进程

```bash
ps -eo pid,etime,pcpu,pmem,cmd | rg 'compute_norm_stats.py|train.py|run_pi05_pickplace_lora.sh'
```

### 查看 checkpoint 目录

```bash
find /nfs_global/S/yangrongzheng/pi05/openpi_official/checkpoints -maxdepth 5 -type f | sort
```

### 查看资产/统计文件

```bash
find /nfs_global/S/yangrongzheng/pi05/openpi_official/assets -maxdepth 5 -type f | sort
```

---

## 7. 当前已知问题

### 7.1 当前环境中的 lerobot 与官方 openpi 有兼容问题

已经做过一次本地 patch，目的是让官方数据集初始化兼容当前环境。

### 7.2 wandb 默认会要求登录

当前训练命令中已经通过：

```bash
--no-wandb-enabled
```

关闭。

### 7.3 如果 checkpoint 目录已经存在

使用：

```bash
--overwrite
```

重新开始训练。

---

## 8. 当前建议

如果你现在只关心官方路线：

1. 确认 `compute_norm_stats` 已跑完
2. 直接用官方命令启动 `pi05_pickplace_lora`
3. 看 `checkpoints/` 目录是否开始持续写入

---

## 9. 关键文件速查

- 原始数据转 LeRobot：`scripts/convert_pick_place_to_lerobot.py`
- 官方 openpi 仓库：`openpi_official/`
- 官方训练脚本：`openpi_official/scripts/train.py`
- 官方 norm stats：`openpi_official/scripts/compute_norm_stats.py`
- 官方自定义配置：`openpi_official/src/openpi/training/pi05_pickplace_config.py`
- 官方训练启动脚本：`openpi_official/run_pi05_pickplace_lora.sh`
