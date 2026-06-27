# PI05 PyTorch LoRA 8GPU 复现使用说明

这份文档面向当前仓库里**已经跑通的 PI05 PyTorch LoRA 训练链路**，对应脚本：`scripts/pi05_pytorch_lora_train_8gpu.slurm`。

已确认的成功样例：

- 任务号：`756045`
- 日志：`logs/pi05-pt-lora-8g-756045.out` / `logs/pi05-pt-lora-8g-756045.err`
- checkpoint 目录：`results/openpi_pytorch_lora_checkpoints/pi05_pickplace_lora_pytorch/pi05_pt_lora_8g_756045`

目前这个链路已经至少成功保存出：

- `1000/`
- `2000/`
- `3000/`

每个目录下都有：

- `model.safetensors`
- `optimizer.pt`
- `metadata.pt`

说明分布式训练、数据读取、模型前向/反向、checkpoint 保存都已经正常工作。

## 1. 这条链路在做什么

这条训练链路是基于 openpi 官方代码路径的 **PI05 pick-place imitation learning + LoRA 微调**。

入口脚本会：

1. 激活本地 conda 环境 `.conda-pi05-openpi-final`
2. source `scripts/use_local_openpi_env.sh`
3. 把本地 LeRobot 数据集挂到 HuggingFace LeRobot 目录下
4. 准备 tokenizer、HF cache、OpenPI cache
5. 用 `srun` 拉起 2 节点 × 每节点 4 卡 = 8 卡分布式训练
6. 在 `openpi_official` 目录里执行：

```bash
python scripts/train_pytorch.py pi05_pickplace_lora_pytorch
```

## 2. 目录约定

当前脚本基本使用固定路径，默认按下面的目录组织运行：

- 仓库根目录：`/nfs_global/S/yangrongzheng/pi05`
- openpi 代码：`/nfs_global/S/yangrongzheng/pi05/openpi_official`
- LeRobot 数据：`/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place`
- 本地 conda 环境：`/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final`
- PI05 初始权重：`/nfs_global/S/yangrongzheng/pi05/pi05_sft`
- RLinf 依赖来源：`/nfs_global/S/yangrongzheng/RLinf-main`
- 输出结果：`/nfs_global/S/yangrongzheng/pi05/results`
- 日志目录：`/nfs_global/S/yangrongzheng/pi05/logs`

## 3. 开始前检查

建议先跑：

```bash
cd /nfs_global/S/yangrongzheng/pi05
bash scripts/check_local_openpi_env.sh
```

建议至少确认下面这些路径存在：

```bash
ls /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final/bin/python
ls /nfs_global/S/yangrongzheng/pi05/openpi_official
ls /nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place
ls /nfs_global/S/yangrongzheng/pi05/pi05_sft
ls /nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages
```

## 4. 如何启动训练

### 4.1 默认提交

```bash
sbatch scripts/pi05_pytorch_lora_train_8gpu.slurm
```

这是当前最直接的复现入口。

### 4.2 自定义实验名

```bash
sbatch --export=ALL,PI05_PT_LORA_EXP_NAME=my_pi05_pt_run scripts/pi05_pytorch_lora_train_8gpu.slurm
```

如果不传，默认实验名是：

```bash
pi05_pt_lora_8g_<jobid>
```

### 4.3 自定义训练步数

```bash
sbatch --export=ALL,PI05_PT_LORA_STEPS=5000 scripts/pi05_pytorch_lora_train_8gpu.slurm
```

默认是：

```bash
PI05_PT_LORA_STEPS=20000
```

### 4.4 自定义 batch size

```bash
sbatch --export=ALL,PI05_PT_LORA_BATCH_SIZE=32 scripts/pi05_pytorch_lora_train_8gpu.slurm
```

脚本传给训练器的是总 batch size，按 `756045` 的实际日志，最终被折算为：

- 每卡 batch size = `2`
- 8 卡总 batch size = `16`

所以如果你改总 batch size，最终是否完全等于设定值，要以训练日志实际打印为准。

### 4.5 自定义精度

```bash
sbatch --export=ALL,PI05_PT_LORA_PRECISION=float32 scripts/pi05_pytorch_lora_train_8gpu.slurm
```

默认是：

```bash
PI05_PT_LORA_PRECISION=bfloat16
```

### 4.6 自定义日志频率和保存频率

```bash
sbatch --export=ALL,PI05_PT_LORA_LOG_INTERVAL=20,PI05_PT_LORA_SAVE_INTERVAL=500 scripts/pi05_pytorch_lora_train_8gpu.slurm
```

默认是：

```bash
PI05_PT_LORA_LOG_INTERVAL=10
PI05_PT_LORA_SAVE_INTERVAL=1000
```

## 5. 如何看日志

### 5.1 查看任务

```bash
squeue -j <jobid>
```

### 5.2 查看 stdout

```bash
tail -f logs/pi05-pt-lora-8g-<jobid>.out
```

### 5.3 查看 stderr

```bash
tail -f logs/pi05-pt-lora-8g-<jobid>.err
```

注意：这个训练链路的**主要训练指标基本都打在 `.err`**，不是 `.out`。

例如：

- `Created experiment checkpoint directory`
- `Using batch size per GPU`
- `step=... loss=... lr=... grad_norm=...`
- `Saved checkpoint ...`

而 `.out` 更多是环境打印、节点信息、Python/JAX 可见性检查。

## 6. 如何判断训练是否正常

至少看下面几件事：

### 6.1 分布式是否正常拉起

`logs/pi05-pt-lora-8g-756045.out` 里能看到：

- `NNODES=2 WORLD_SIZE=8`
- 两台机器的 rank 都拉起来了
- 每个 rank 都能看到同一个 Python 环境
- `jax_ok`
- `NCCL version ...`

这说明多机多卡初始化是正常的。

### 6.2 数据和归一化统计是否加载成功

`logs/pi05-pt-lora-8g-756045.err` 里能看到：

- `Loaded norm stats from ...`
- `data_config: DataConfig(...)`

这说明数据资产和归一化配置已经被训练器读到。

### 6.3 训练步是否持续推进

`756045` 的日志里已经推进到至少 `step=3940`，并且中间没有出现 traceback 或 rank 崩溃。

### 6.4 checkpoint 是否按间隔落盘

目前已经确认存在：

```text
results/openpi_pytorch_lora_checkpoints/pi05_pickplace_lora_pytorch/pi05_pt_lora_8g_756045/1000
results/openpi_pytorch_lora_checkpoints/pi05_pickplace_lora_pytorch/pi05_pt_lora_8g_756045/2000
results/openpi_pytorch_lora_checkpoints/pi05_pickplace_lora_pytorch/pi05_pt_lora_8g_756045/3000
```

这通常说明训练主循环和保存逻辑都正常。

## 7. `756045` 日志是否正常

结论：**目前看是正常的，而且已经算是明确跑通了。**

主要依据：

1. 8 个 rank 都正常启动。
2. 没有看到 `Traceback`、NCCL fatal error、CUDA OOM、dataset crash。
3. loss 在持续打印，step 在持续增长。
4. 已经连续保存出多个 checkpoint。
5. 训练速度稳定在每 10 step 约 7~8 秒量级，没有明显卡死。

目前唯一比较显眼的“非训练错误”是这类提示：

```text
Error while loading conda entry point: conda-anaconda-tos (No module named 'pydantic_core._pydantic_core')
```

这个报错在 `756045` 中出现了多次，但它没有阻止 conda 激活，也没有阻止训练启动；从后续训练完整进行来看，**它更像是 conda 插件告警，而不是训练本身故障**。

如果你想把日志弄干净，可以后面单独处理这个插件问题；但就当前任务是否跑通而言，它不是 blocker。

## 8. 为什么 loss 波动这么大

结论先说：**从当前日志看，这种波动更像是 step-level 的正常抖动，不像训练已经发散。**

`756045` 里你能看到单步 loss 经常在下面区间跳动：

- 低的时候到 `0.004` ~ `0.01`
- 常见在 `0.02` ~ `0.06`
- 偶尔到 `0.07` ~ `0.09`

但每 10 step 聚合打印出来的 loss 大致仍在 `0.03` ~ `0.04` 左右，例如：

- `step=3900 loss=0.0436 grad_norm=0.11`
- `step=3910 loss=0.0381 grad_norm=0.12`
- `step=3930 loss=0.0371 grad_norm=0.14`
- `step=3940 loss=0.0361 grad_norm=0.12`

也就是说：

- **单 step loss 抖动明显**
- **窗口平均 loss 其实比较稳**
- **grad_norm 也比较稳（约 0.11 ~ 0.14）**

这通常说明训练在正常收敛区间内，只是 mini-batch 噪声较大。

### 造成波动大的常见原因

#### 8.1 每卡 batch 太小

日志显示：

- 每卡 batch size = `2`
- 8 卡总 batch size = `16`

每卡 batch 很小时，step-level loss 本来就会很抖，尤其是机器人模仿学习这种样本异质性较强的数据。

#### 8.2 数据本身异质性强

pick-place 数据里，不同阶段样本难度差异通常很大，例如：

- 接近目标前的细粒度控制
- 抓取瞬间
- 抬升和放置阶段
- 图像状态变化幅度不同

如果某个 mini-batch 刚好抽到更难的片段，loss 会明显抬高。

#### 8.3 LoRA 微调只改一部分参数

LoRA 更新参数量小，适配初期容易表现为：

- 某些 batch 很容易拟合
- 某些 batch 很难拟合
- step-to-step loss 抖动比全参训练更明显

#### 8.4 当前看的是瞬时 loss，不是平滑曲线

你现在看到的大多是 tqdm 上的瞬时 step loss，这个值天然噪声就大。
真正更值得看的是：

- 每 N step 的聚合 loss
- checkpoint 间评测表现
- rollout 成功率 / 成功轨迹比例

## 9. 什么情况下才算“不正常”

如果后续日志出现下面这些现象，才更值得担心：

- 聚合 loss 持续单调升高，而不是围绕某个范围波动
- `grad_norm` 持续飙升到很高，并且越来越大
- 经常出现 `nan` / `inf`
- rank 随机退出，出现 NCCL timeout / CUDA OOM
- 保存 checkpoint 失败
- 训练速度越来越慢，甚至卡死

就 `756045` 当前看到的片段来说，还没有这些坏信号。

## 10. 建议怎么继续用

### 方案 A：按当前配置继续跑

如果目标是先证明“链路可复现、能稳定出 checkpoint”，当前配置已经够用了。

### 方案 B：为了让 loss 更平滑，优先尝试这几个方向

1. 提高有效 batch size
2. 降低学习率
3. 拉长观察窗口，不盯单 step loss
4. 增加固定 checkpoint 间隔下的评测

例如可以尝试：

```bash
sbatch --export=ALL,PI05_PT_LORA_BATCH_SIZE=64 scripts/pi05_pytorch_lora_train_8gpu.slurm
```

或者：

```bash
sbatch --export=ALL,PI05_PT_LORA_PRECISION=float32 scripts/pi05_pytorch_lora_train_8gpu.slurm
```

不过是否值得改，要看你的目标：

- 如果只是复现跑通：现在已经够好
- 如果要追求更稳的曲线和更好的最终效果：再做 batch / lr / eval 调参

## 11. 推荐的日常命令

```bash
cd /nfs_global/S/yangrongzheng/pi05
bash scripts/check_local_openpi_env.sh
sbatch scripts/pi05_pytorch_lora_train_8gpu.slurm
squeue -j <jobid>
tail -f logs/pi05-pt-lora-8g-<jobid>.out
```

重点盯训练指标时，更推荐：

```bash
tail -f logs/pi05-pt-lora-8g-<jobid>.err
```

## 12. 后续可以补什么

如果后面你想把这条链路做得更完整，建议再补三样：

1. checkpoint 自动评测脚本
2. loss / grad_norm 曲线导出脚本
3. 从 checkpoint 恢复训练的 README 示例

这样就不只是“能跑通”，而是“能持续复现实验并做对比”。
