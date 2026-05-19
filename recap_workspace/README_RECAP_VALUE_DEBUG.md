# RECAP Value Debug 状态说明

这份文档描述当前 `recap_workspace` 下 RECAP value 训练链路的可用用法、临时调试改动，以及后续需要完成的清理和正式化工作。

## 当前结论

目前已经**跑通了 debug 链路**，含义是：

- Slurm 脚本可以正常提交
- Ray head 可以正常启动
- Hydra 配置可以正确加载
- `ActorGroup` 可以创建成功
- `runner.init_workers()` 可以进入
- `ValueDataset` / `DataLoader` 可以开始构建
- `ValueCriticModel` 可以开始初始化
- `runner.run()` 已经进入

这说明从“脚本起不来 / actor 崩 / 配置错 / sidecar 缺失”已经推进到“真实训练链路可执行”。

## 当前推荐用法

### 1. 提交 debug job

在仓库根目录执行：

```bash
sbatch recap_workspace/scripts/recap_value_debug_1gpu.slurm
```

这个脚本目前用于**链路验证**，不是正式训练脚本。

### 2. 查看主日志

```bash
tail -f recap_workspace/logs/recap-value-dbg1-<jobid>.out
tail -f recap_workspace/logs/recap-value-dbg1-<jobid>.err
```

### 3. 查看 worker 日志

每次 job 会写独立目录：

```bash
recap_workspace/logs/recap_value_<jobid>/worker_logs/ActorGroup/rank_0.log
```

如果链路正常推进，通常会看到：

- `build_dataloader ...`
- `data iterator ready`
- `Creating ValueCriticModel`
- `setup_model_and_optimizer done`

## 当前为了“先跑通链路”做过的临时改动

以下改动是**调试态**措施，不一定适合直接保留到正式训练：

### 1. `run_value_local.py` 里的 runtime patch

文件：`recap_workspace/run_value_local.py`

当前做了这些事：

- 强制把 HuggingFace datasets cache 指向可写目录
- patch `LeRobotDataset._query_hf_dataset`
- patch `lerobot` 的 timestamp sync 检查，遇到异常时先放过
- 在主入口结束时用 `os._exit()` 绕过解释器 teardown 阶段的 segfault

### 2. `sft_runner.py` 初始化顺序修正

文件：`RLinf-recap/rlinf/runners/sft_runner.py`

已修正：

- 不再在 `SFTRunner.__init__()` 里提前调用 `set_max_steps()`
- 改为在 `init_workers()` 里，`actor.init_worker().wait()` 之后再计算 `max_steps`

这属于**真正的逻辑修复**，建议保留。

### 3. `fsdp_value_sft_worker.py` 调试期改动

文件：`RLinf-recap/rlinf/workers/sft/fsdp_value_sft_worker.py`

当前包含：

- 额外日志
- 将 dataloader 初始化延后到 `init_worker()`
- 调试异常打印

其中“日志”和“异常打印”是调试用；
“是否延后 dataloader 初始化”需要后续判断是否作为正式修复保留。

## 当前 debug 使用的数据

### 临时最小数据集

为了快速跑通链路，目前配置指向：

```bash
/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_mini
```

这是从原始数据集中裁出来的一个**最小测试集**，只保留少量 episode，用于缩短 dataset materialization 时间。

### 临时 sidecar 文件

为了先把 value 训练链路接起来，还生成了一个**伪造的** returns sidecar：

```bash
/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_mini/meta/returns_local_pi05.parquet
```

以及原始大数据目录下也补过一个临时版本：

```bash
/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place/meta/returns_local_pi05.parquet
```

注意：这两个 sidecar 只是为了**测试链路**，不代表真实 value 标签。

## 当前配置状态

文件：`recap_workspace/configs/local_value_sft.yaml`

当前是 debug 配置，特点包括：

- 训练数据指向 `lerobot_pick_place_mini`
- `train_data_paths[0].max_samples = 64`
- `eval_data_paths[0].max_samples = 64`
- `runner.max_steps=1` / `runner.max_epochs=1` 是通过 Slurm override 注入
- 显式设置了：
  - `data.return_min: -1109.0`
  - `data.return_max: 0.0`

## 如果你现在要继续使用

### 适合做的事

当前适合：

- 验证启动链路
- 验证模型初始化链路
- 验证 dataset / sidecar / Ray / Hydra 是否工作
- 做 smoke test

### 不适合直接做的事

当前**不建议直接拿来正式训练**，因为：

- 使用的是 mini 数据集
- sidecar 是伪造的
- timestamp 校验被绕过了
- 入口里保留了调试性质的强制退出和 patch

## 后续要做的事情

### A. 恢复真实数据

需要把配置从：

```bash
/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_mini
```

切回：

```bash
/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place
```

### B. 生成真实 returns sidecar

需要用正式流程生成：

```bash
meta/returns_local_pi05.parquet
```

而不是继续使用临时伪造版本。

### C. 决定如何处理 timestamp sync 问题

当前是通过 runtime patch 忽略时间戳不同步异常。

后续需要二选一：

- 修数据本身的时间戳问题
- 或确认该检查对当前数据不必要，并以可维护方式关闭/放宽它

### D. 清理调试 patch

后续需要检查哪些 patch 是正式修复，哪些只是 debug 用：

建议保留：

- `SFTRunner` 初始化顺序修复

建议评估后清理或重构：

- `run_value_local.py` 中的 timestamp patch
- `run_value_local.py` 中的 `os._exit()`
- `fsdp_value_sft_worker.py` 中的额外日志和调试异常打印
- mini 数据集和伪造 sidecar

### E. 恢复更真实的训练规模

等真实数据恢复后，再逐步恢复：

- 完整数据集
- 更大的 `max_samples`
- 更真实的 `runner.max_steps`
- 正式 batch size / eval size

## 快速检查清单

如果后面有人接手，可以按下面顺序排查：

1. `sbatch recap_workspace/scripts/recap_value_debug_1gpu.slurm`
2. 看 `recap-value-dbg1-<jobid>.out/.err`
3. 看 `recap_value_<jobid>/worker_logs/ActorGroup/rank_0.log`
4. 确认是否出现：
   - `build_dataloader done`
   - `data iterator ready`
   - `Creating ValueCriticModel`
   - `setup_model_and_optimizer done`
   - `runner.run start`
5. 如果要做正式训练，先恢复真实 sidecar 和真实数据

## 一句话总结

现在这套东西已经可以用于：

- **验证 RECAP value 训练链路本身是通的**

但要进入正式训练，还需要：

- **恢复真实数据**
- **生成真实 returns sidecar**
- **处理时间戳同步问题**
- **清理临时 debug patch**

## Online data integration

To train ReCap value on both the original offline dataset and newly collected real-robot online data:

1. Collect raw real-robot rollouts:
   `python scripts/pi05_recap_real_collect.py --checkpoint-dir <ckpt> --prompt "..." --num-episodes 1`
2. Convert them into LeRobot format:
   `python scripts/convert_pi05_rollout_to_lerobot.py --overwrite`
3. Generate sparse returns sidecar:
   `python scripts/generate_sparse_returns.py --dataset-dir /nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_online --tag local_pi05_online`
4. Launch mixed offline+online value training:
   `sbatch recap_workspace/scripts/recap_value_train_online_1gpu.slurm`

The mixed config lives at `recap_workspace/configs/local_value_sft_online.yaml`.
