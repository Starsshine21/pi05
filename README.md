# PI05 / RECAP 使用说明

这份 `README.md` 只保留三部分最重要的内容：

1. 如何配置当前仓库环境
2. 如何跑 PI05 的真机推理
3. 如何跑 RECAP 的真机强化学习

默认前提：

- 你会自己准备好 `pi05` 的 policy model checkpoint
- 你会自己准备好 RECAP 所需的 value model
- 本仓库只负责代码、环境、脚本和流程，不包含日志、训练数据和大模型文件

---

## 1. 如何配置环境

这一部分的目标是：

- 在一台新机器上把当前仓库的运行环境搭起来
- 能成功 import `openpi_official`
- 能成功 import `recap_workspace` 依赖链路
- 能加载 PI05 checkpoint 与 RECAP value 训练入口
- 能继续跑真机推理或 RECAP 真机采集 / 训练

### 1.1 当前目录约定

当前仓库已经包含运行 RECAP 和 PI05 所需的本地代码布局，默认使用下面这些目录：

- `openpi_official/`：PI05 policy 训练与推理代码
- `recap_workspace/`：RECAP value 训练与调试入口
- `recap_workspace/pi06_recap/`：PI06 / RECAP value、label、advantage 逻辑实现
- `recap_workspace/vendor/`：RECAP 运行时代码
- `local_vendor/`：当前仓库内保留的补充配置与本地依赖副本
- `models/openpi/big_vision/`：tokenizer 等 OpenPI 运行资源
- `models/huggingface/lerobot/`：本地 LeRobot 数据入口
- `data/`：本地数据目录
- `.cache/`：HuggingFace / OpenPI / matplotlib 等缓存目录

也就是说，**现在的环境默认只依赖当前 `pi05/` 目录本身**。

---

### 1.2 创建 Python 环境

推荐使用 Python 3.11。

如果你要新建一个干净环境：

```bash
conda create -n pi05 python=3.11 -y
conda activate pi05
```

如果你沿用当前仓库已有环境，也可以直接激活本地环境，例如：

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
```

只要后续命令里的 `python` 指向你实际想用的环境即可。

---

### 1.3 安装 PI05 / OpenPI 依赖

先安装 `openpi_official`：

```bash
cd openpi_official
pip install -e .
cd ..
```

再安装当前仓库补充依赖：

```bash
pip install -r requirements-standalone.txt
```

如果你要跑 `recap_workspace`，还需要安装 RECAP 的最小依赖：

```bash
pip install -r recap_workspace/requirements_recap_min.txt
```

如果你已经有一套可用环境，也可以只补装缺失包。

---

### 1.4 加载仓库环境变量

当前主环境脚本是：

```bash
scripts/use_local_openpi_env.sh
```

使用方式：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/your/env
source scripts/use_local_openpi_env.sh
```

这个脚本会默认完成几件事：

- 把 `openpi_official/src` 加入 `PYTHONPATH`
- 把 `recap_workspace/vendor/` 下的 RECAP 运行时代码加入 `PYTHONPATH`
- 把本地 `libero` 目录加入 `PYTHONPATH`
- 设置 `HF_HOME`、`HF_DATASETS_CACHE`、`OPENPI_DATA_HOME`
- 设置 `MPLCONFIGDIR`
- 设置本地 tokenizer、LeRobot 数据、CUDA runtime 的搜索路径
- 导出 `RECAP_WORKSPACE`、`EMBODIED_PATH` 等仓库内部默认路径

正常情况下，你不需要再手动配置这些变量。

---

### 1.5 环境检查

如果你想先确认 `openpi_official` 环境：

```bash
python scripts/check_local_openpi_env.py
```

如果你想确认 `recap_workspace` 的导入链路：

```bash
python recap_workspace/import_probe.py
```

这两个检查都通过后，通常说明当前目录已经具备：

- PI05 policy 推理代码可导入
- LeRobot 数据接口可导入
- RECAP value worker 相关模块可导入

---

### 1.6 真机相关额外依赖

如果你只做离线加载 checkpoint，不一定需要真机依赖。

如果你要跑 PI05 真机推理或 RECAP 真机采集，目标机还需要：

- `rtde_control`
- `rtde_receive`
- `pyserial`
- `pyrealsense2`
- `pyorbbecsdk`

对应硬件含义：

- `rtde_control` / `rtde_receive`：UR5e 机械臂
- `pyserial`：Inspire hand
- `pyrealsense2`：L515
- `pyorbbecsdk`：Orbbec Femto Bolt

---

### 1.7 你自己需要准备的模型

这个仓库不包含模型文件，所以你需要自己额外准备：

#### PI05 policy model

你需要准备一个可推理的 PI05 checkpoint 目录，例如：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

推理至少需要：

- `model.safetensors`
- `metadata.pt`
- `assets/local/pi05-pickplace-il/norm_stats.json`

#### RECAP value model

你需要准备可用的 RECAP value model 或已有 checkpoint。

本仓库只假设：

- 你已经有可用的 value model 路径
- 你会在 RECAP 运行命令或配置中显式指定它

---

## 2. 如何跑 PI05 的真机推理

这一部分只讲最小可用的 PI05 真机推理链路。

### 2.1 入口脚本

PI05 真机推理脚本是：

`scripts/pi05_real_robot_infer.py`

它会完成：

- 连接 UR5e
- 连接 Inspire hand
- 打开 L515
- 打开 Orbbec Femto Bolt
- 读取 observation
- 加载 `openpi_official` policy
- 输出 action
- 下发到机械臂和手爪

### 2.2 当前 observation 定义

推理输入是：

- `image`：来自 L515
- `wrist_image`：来自 Orbbec Femto Bolt
- `state`：`concat([joints, eef, hand])`，18 维
- `prompt`：文本任务描述

### 2.3 当前 action 解释方式

当前脚本把输出动作解释为：

- 前 6 维：joint delta
- 后 6 维：hand delta

也就是：

```python
target_joints = current_joints + action[:6]
target_hand = current_hand + action[6:12]
```

### 2.4 启动前你需要确认

- 目标机已经装好真机相关 Python 包和 SDK
- 你已经有可用 PI05 checkpoint
- 你知道机器人 IP
- 你知道 Inspire hand 的串口路径

### 2.5 启动命令

推荐直接使用当前仓库环境：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/your/env
source scripts/use_local_openpi_env.sh

python scripts/pi05_real_robot_infer.py \
  --checkpoint-dir /path/to/pi05_model_checkpoint/60000 \
  --train-config pi05_pickplace_full_pytorch \
  --prompt "place the cookie at the left top position on the desk" \
  --robot-ip 192.168.1.109 \
  --hand-port /dev/ttyUSB0
```

### 2.6 首次上真机建议

第一次不要直接高速跑，建议把参数压低：

```bash
python scripts/pi05_real_robot_infer.py \
  --checkpoint-dir /path/to/pi05_model_checkpoint/60000 \
  --train-config pi05_pickplace_full_pytorch \
  --prompt "place the cookie at the left top position on the desk" \
  --robot-ip 192.168.1.109 \
  --hand-port /dev/ttyUSB0 \
  --control-hz 5 \
  --arm-speed 0.03 \
  --arm-acceleration 0.03
```

### 2.7 当前真机链路的边界

现在这条 PI05 真机链路是最小可用版，已经能跑，但还不是完全工程化版本。

当前还缺：

- 关节限位保护
- 工作空间边界保护
- 碰撞保护
- `dry-run` 模式
- 启动前自检脚本
- 更细的异常恢复

所以建议：

- 先做人工盯守下的小步验证
- 不要直接长时间无保护运行

---

## 3. 如何跑 RECAP 的真机强化学习

这里的“RECAP 真机强化学习”按当前仓库里的最小工作流来理解：

1. 用已有 PI05 policy 在真机上采集 rollout
2. 生成后续 value / return / label 所需输入
3. 使用你自己准备好的 value model 继续后处理、打分或训练

### 3.1 当前 RECAP 相关目录

你主要会用到这些位置：

- `scripts/pi05_recap_real_collect.py`
- `recap_workspace/`
- `recap_workspace/pi06_recap/`

### 3.2 真机采集入口脚本

真机 rollout 采集脚本是：

`scripts/pi05_recap_real_collect.py`

这个脚本本质上会：

- 加载 PI05 policy checkpoint
- 读取真机 observation
- 执行动作
- 把每一帧保存下来
- 保存每个 episode 的 `frames.npz`、`meta.json`、`thumb.png`

### 3.3 采集输出默认目录

默认输出目录是：

```bash
data/recap_real_collect
```

你也可以显式指定：

```bash
--output-dir /path/to/recap_real_collect
```

### 3.4 采集命令

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate /path/to/your/env
source scripts/use_local_openpi_env.sh

python scripts/pi05_recap_real_collect.py \
  --checkpoint-dir /path/to/pi05_model_checkpoint/60000 \
  --train-config pi05_pickplace_full_pytorch \
  --prompt "place the cookie at the left top position on the desk" \
  --output-dir /path/to/recap_real_collect \
  --robot-ip 192.168.1.109 \
  --hand-port /dev/ttyUSB0 \
  --num-episodes 1 \
  --max-steps 100
```

常用参数：

- `--episode-start-index`：从哪个 episode 编号开始
- `--num-episodes`：采多少条 episode
- `--max-steps`：每条轨迹最大步数
- `--overwrite`：覆盖已有输出目录

### 3.5 采集后的数据格式

每个 episode 目录下通常会有：

- `frames.npz`
- `meta.json`
- `thumb.png`

这些是后续 RECAP 处理的原始输入。

### 3.6 后续工作流入口

当前仓库里的 RECAP 代码主要在：

- `recap_workspace/`
- `recap_workspace/pi06_recap/`

按当前代码结构，后续通常会分几步：

1. 基于 rollout 生成 returns / sparse returns / labels
2. 准备 value model 训练或打分输入
3. 跑 value model / score / advantage 相关脚本
4. 再把结果并回后续工作流

### 3.7 你需要重点看的文件

#### `recap_workspace/`

这个目录更偏工作流和配置：

- `recap_workspace/README.md`
- `recap_workspace/docs/USAGE.md`
- `recap_workspace/run_value_local.py`
- `recap_workspace/patch_lerobot_runtime.py`
- `recap_workspace/configs/*.yaml`
- `recap_workspace/scripts/*.slurm`

#### `recap_workspace/pi06_recap/`

这个目录更偏 value / recap 逻辑实现：

- `recap_workspace/pi06_recap/train_vf.py`
- `recap_workspace/pi06_recap/score_vf.py`
- `recap_workspace/pi06_recap/advantage.py`
- `recap_workspace/pi06_recap/labels.py`
- `recap_workspace/pi06_recap/manifest.py`
- `recap_workspace/pi06_recap/vf_data.py`
- `recap_workspace/pi06_recap/vf_model.py`

### 3.8 当前这部分怎么理解

如果你已经准备好了：

- 可用的 PI05 checkpoint
- 可用的 RECAP value model 或已有 value checkpoint

那么当前最实际的工作流就是：

1. 先把环境跑通
2. 先让 PI05 真机 policy 能跑
3. 用 `scripts/pi05_recap_real_collect.py` 在真机上采集 rollout
4. 再把 rollout 喂给 `recap_workspace/` 和 `recap_workspace/pi06_recap/` 里的脚本
5. 用你自己的 value model 继续做后续打分、训练或强化学习流程

### 3.9 当前 RECAP 链路的边界

需要明确一点：

- 当前仓库里已经有 RECAP 相关代码和真机采集入口
- 但完整自动化的一键真机 RL pipeline 还没有整理成单条命令

也就是说，现在已经具备：

- 真机 rollout 采集
- value function / returns / labels / advantage 相关实现
- 本地 / slurm 运行脚本

但你在真正跑 RECAP 流程时，仍然需要根据自己的：

- value model 路径
- rollout 数据位置
- 目标训练策略
- 本地还是集群运行方式

做少量参数拼接。

---

## 最后建议的实际顺序

如果你现在要开始干活，我建议按这个顺序：

### 第一步：先把环境跑通

```bash
conda create -n pi05 python=3.11 -y
conda activate pi05
cd openpi_official && pip install -e . && cd ..
pip install -r requirements-standalone.txt
pip install -r recap_workspace/requirements_recap_min.txt
source scripts/use_local_openpi_env.sh
python scripts/check_local_openpi_env.py
python recap_workspace/import_probe.py
```

### 第二步：先跑 PI05 真机推理

确认：

- PI05 checkpoint 可加载
- UR5e / hand / camera 都能连上
- 真机 observation 正常

### 第三步：再跑 RECAP 真机采集

先用 `scripts/pi05_recap_real_collect.py` 采几条短 rollout。

### 第四步：再接 RECAP value model

确认 rollout 存储格式没问题后，再进入：

- `recap_workspace/`
- `recap_workspace/pi06_recap/`

继续做 returns / value / advantage / training。
