# PI05 环境迁移与真机推理文档

本文档基于当前目录 `/nfs_global/S/yangrongzheng/pi05` 的实际状态整理，目标是：

1. 说明当前已经训练完成的 PI05 PyTorch Full 模型保存在哪里。
2. 检查当前“真机推理链路”是否已经完整。
3. 给出把当前环境迁移到另一台电脑上，并运行 PI05 真机推理的步骤。

## 1. 当前训练好的 PI05 模型保存位置

根据日志 `logs/pi05-pickplace-pt-full-757027.err`：

- 作业启动时创建的实验目录是：

```bash
/nfs_global/S/yangrongzheng/pi05/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027
```

日志中能看到：

- `Created experiment checkpoint directory: /nfs_global/S/yangrongzheng/pi05/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027`

该实验目录下已经存在多个 step checkpoint，包括：

- `1000`
- `2000`
- ...
- `59000`
- `59999`
- `60000`

如果你要拿“当前这次 full train 的最终模型”做推理，优先使用：

```bash
/nfs_global/S/yangrongzheng/pi05/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

该目录下包含至少这些关键文件：

- `model.safetensors`
- `metadata.pt`
- `optimizer.pt`
- `assets/local/pi05-pickplace-il/norm_stats.json`

其中推理真正必需的核心是：

- `model.safetensors`
- `metadata.pt`
- `assets/local/pi05-pickplace-il/norm_stats.json`

建议迁移时直接拷整个 `60000/` 目录，最稳妥。

## 2. 当前真机链路是否已经完整

结论先说：

- **软件侧主链路基本已经打通，可以形成“加载 checkpoint → 采集真机观测 → 输出动作 → 下发机械臂/手爪”的最小闭环。**
- **但它目前仍然是“最小可用版”，距离“可安全稳定上线真机长期跑任务”的完整工程链路还有缺口。**

## 3. 已经具备的链路

当前仓库里已经有：

- 真机推理脚本：`scripts/pi05_real_robot_infer.py`
- 真机接入说明：`README_PI05_REAL_ROBOT.md`
- 训练配置：`pi05_pickplace_full_pytorch`
- 已训练完成 checkpoint：`results/openpi_official_pytorch_full_checkpoints/.../60000`
- 环境注入脚本：`scripts/use_local_openpi_env.sh`

### 3.1 输入链路已具备

`scripts/pi05_real_robot_infer.py` 已实现以下 observation 采集：

- `image`: 来自 `L515`
- `wrist_image`: 来自 `Orbbec Femto Bolt`
- `state`: `concat([joints, eef, hand])`，共 18 维
- `prompt`: 外部传入文本

这与当前 README 中描述的训练数据 schema 是一致的。

### 3.2 模型加载链路已具备

脚本里通过：

- `from openpi.training import config as train_config_lib`
- `from openpi.policies import policy_config as policy_config_lib`

以及：

```python
train_cfg = train_config_lib._CONFIGS_DICT[train_config_name]
self.policy = policy_config_lib.create_trained_policy(train_cfg, pathlib.Path(checkpoint_dir), default_prompt=prompt)
```

完成 checkpoint 的加载。

这意味着只要：

- `openpi_official/src` 在 `PYTHONPATH` 中
- checkpoint 目录完整
- norm stats 目录齐全

就能进入 policy inference。

### 3.3 执行链路已具备

脚本中已经封装了三类硬件接口：

- `UR5eRTDE`
- `InspireHandSerial`
- `L515ColorCamera`
- `OrbbecFemtoBoltColorCamera`

推理输出当前解释为：

- 前 6 维：`joint delta`
- 后 6 维：`hand delta`

并执行：

- `target_joints = current_joints + action[:6]`
- `target_hand = current_hand + action[6:12]`

然后下发到：

- UR5e RTDE
- Inspire Hand serial

所以从“代码可运行性”来看，闭环是完整的。

## 4. 当前链路的缺口与风险

虽然主链已通，但我不建议把现在状态直接理解成“真机链路已经完全工程化”。主要缺口如下。

### 4.1 缺少安全边界

当前 `scripts/pi05_real_robot_infer.py` 中没有看到：

- 关节限位保护
- 工作空间边界保护
- 自碰撞/桌面碰撞保护
- 速度/加速度二次裁剪
- 紧急停止逻辑
- 首帧确认/人工授权机制

这意味着只要模型输出异常，动作会直接变成目标关节命令，存在真机风险。

### 4.2 动作语义是“当前假设”，并非经过真机闭环充分验证

README 明确写了：

- 当前脚本按 `joint delta + hand delta` 解释模型输出。

这与训练数据设计是一致的，但还需要确认：

- 训练时 action 是否就是“机器人当前时刻下的 joint delta / hand delta”
- hand 的 6 维范围和真机控制协议是否完全匹配
- 模型输出尺度是否和真机执行频率 `10Hz` 匹配

也就是说：

- **训练链语义对齐基本成立**
- **但真机稳定性仍需要实际 dry-run 验证**

### 4.3 缺少设备自检/降级逻辑

脚本当前没有完整做：

- 相机打开失败后的友好报错和降级
- 手爪串口不存在时的预检查
- 机器人网络不可达时的预检查
- observation shape / dtype 的启动前校验
- checkpoint 与 config 不匹配时的显式检查

### 4.4 迁移依赖仍然偏“路径耦合”

当前环境强依赖：

- `.conda-pi05-openpi-final`
- `scripts/use_local_openpi_env.sh`
- `/nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages`
- `/nfs_global/S/yangrongzheng/RLinf-main/RLinf_deps/libero`
- 以及若干 CUDA 动态库目录

也就是说，**当前环境并不是一个完全自包含、可直接 rsync 后无脑运行的环境**。

## 5. 结论：真机链路完整性判断

我的判断是：

- **从“研究复现/实验验证”标准看：已经基本完整。**
- **从“稳定部署到另一台电脑直接上真机”标准看：还不完全完整，需要把环境依赖和硬件依赖显式补齐。**

更准确地说，现在已经具备：

- 数据 schema 对齐
- 官方 openpi policy 加载
- 双相机 + 机器人 + 手爪采集执行
- 最小在线 rollout loop

但还缺：

- 安全保护层
- 迁移环境去路径耦合
- 真机上线前自检脚本
- 明确的依赖清单与安装脚本

## 6. 迁移到另一台电脑时，最少需要带走什么

建议至少迁移以下内容。

### 6.1 仓库本体

```bash
/nfs_global/S/yangrongzheng/pi05
```

### 6.2 推理所需 checkpoint

至少带走：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

### 6.3 openpi 代码

当前仓库里已经包含：

```bash
openpi_official/
```

推理依赖它的：

- `src/openpi/...`
- config
- policy loader

### 6.4 环境描述文件

建议一起带走：

- `.conda-pi05-openpi-explicit`
- `openpi_official/pyproject.toml`
- `openpi_official/uv.lock`
- `scripts/use_local_openpi_env.sh`

### 6.5 可能还要带走的外部依赖来源

从当前脚本看，目标机还要满足：

- `rtde_control`
- `rtde_receive`
- `pyserial`
- `pyrealsense2`
- `pyorbbecsdk`
- `opencv-python`
- `torch`
- `transformers`
- `lerobot`
- `openpi_official` 依赖链

而当前 `scripts/use_local_openpi_env.sh` 还依赖：

- `/nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages`
- `/nfs_global/S/yangrongzheng/RLinf-main/RLinf_deps/libero`

如果目标机器上没有这些路径，需要改造这个脚本。

## 7. 推荐的迁移方式

我推荐两种方式，优先推荐第一种。

### 方案 A：直接打包当前 conda 环境（推荐）

如果目标机器系统足够接近（Linux 发行版、CUDA 驱动、Python ABI 接近），最稳妥的方式是：

1. 打包当前 conda 环境 `.conda-pi05-openpi-final`
2. 拷贝仓库目录
3. 拷贝 checkpoint `60000/`
4. 在目标机重新安装硬件 SDK（RealSense / Orbbec / RTDE）
5. 修改 `scripts/use_local_openpi_env.sh` 中的硬编码路径

优点：

- 最容易接近当前可运行状态

缺点：

- 目录与驱动版本耦合较强

### 方案 B：在目标机按 lockfile 重建环境

使用：

- `openpi_official/pyproject.toml`
- `openpi_official/uv.lock`
- `.conda-pi05-openpi-explicit`

重建 Python 环境，再补装真机 SDK。

优点：

- 更干净，更适合长期维护

缺点：

- 重建成本更高
- 对依赖版本更敏感

## 8. 目标机环境复现步骤（推荐流程）

以下步骤默认目标机是 Linux，且有 NVIDIA 驱动，Python 3.11，可连接 UR5e、Inspire hand、L515、Femto Bolt。

### 第一步：复制仓库

```bash
rsync -av /nfs_global/S/yangrongzheng/pi05 <TARGET_HOST>:<TARGET_PARENT>/
```

### 第二步：复制最终 checkpoint

如果仓库已整体复制，这一步可省略；否则至少复制：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

### 第三步：准备 Python 环境

优先尝试复用当前环境名：

```bash
conda create -p <TARGET_PARENT>/pi05/.conda-pi05-openpi-final python=3.11 -y
conda activate <TARGET_PARENT>/pi05/.conda-pi05-openpi-final
```

然后按需要安装：

- PyTorch
- OpenCV
- Transformers
- LeRobot
- openpi_official 依赖
- pyserial
- ur RTDE Python 包
- pyrealsense2
- pyorbbecsdk

如果你能直接导出/复制当前环境，会更省事。

### 第四步：修正环境脚本

修改：

```bash
scripts/use_local_openpi_env.sh
```

至少检查这些变量：

- `PI05_CONDA_ENV_ROOT`
- `RLINF_ROOT`
- `RLINF_VENV_ROOT`
- `RLINF_VENV_SITE_PACKAGES`
- `LIBERO_REPO_PATH`
- `HF_LEROBOT_HOME`

如果目标机没有 `RLinf-main`，建议改成：

- 删除对 `RLinf-main/.venv/site-packages` 的依赖
- 直接使用目标机本地 conda 环境安装完整依赖
- 仅保留 `PYTHONPATH=$REPO_ROOT/openpi_official/src:$PYTHONPATH`

### 第五步：验证 Python 侧依赖

在目标机运行：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate <TARGET_PARENT>/pi05/.conda-pi05-openpi-final
source <TARGET_PARENT>/pi05/scripts/use_local_openpi_env.sh

python - <<'PY'
import cv2
import numpy
import torch
import serial
import pyrealsense2
print('python deps ok')
PY
```

然后再检查：

```bash
PYTHONPATH=<TARGET_PARENT>/pi05/openpi_official/src:$PYTHONPATH \
python - <<'PY'
from openpi.training import config as train_config_lib
from openpi.policies import policy_config as policy_config_lib
print('openpi import ok')
print('config exists:', 'pi05_pickplace_full_pytorch' in train_config_lib._CONFIGS_DICT)
PY
```

### 第六步：验证 checkpoint 可加载

```bash
cd <TARGET_PARENT>/pi05
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate <TARGET_PARENT>/pi05/.conda-pi05-openpi-final
source scripts/use_local_openpi_env.sh

PYTHONPATH=<TARGET_PARENT>/pi05/openpi_official/src:$PYTHONPATH \
python scripts/pi05_real_robot_infer.py \
  --checkpoint-dir <TARGET_PARENT>/pi05/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000 \
  --train-config pi05_pickplace_full_pytorch \
  --prompt "place the cookie at the left top position on the desk" \
  --robot-ip 192.168.1.109 \
  --hand-port /dev/ttyUSB0
```

第一次建议不要真正连真机执行，先确认：

- import 不报错
- checkpoint 能成功加载
- 相机能打开
- 手爪串口能打开
- 机器人 RTDE 能连上

## 9. 真机推理步骤

当前最小可用命令就是 README 中这条，实际应把 `<EXP_NAME>` 替换成现有实验名：

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
source /nfs_global/S/yangrongzheng/pi05/scripts/use_local_openpi_env.sh

PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/openpi_official/src:$PYTHONPATH \
python scripts/pi05_real_robot_infer.py \
  --checkpoint-dir /nfs_global/S/yangrongzheng/pi05/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000 \
  --train-config pi05_pickplace_full_pytorch \
  --prompt "place the cookie at the left top position on the desk" \
  --robot-ip 192.168.1.109 \
  --hand-port /dev/ttyUSB0
```

## 10. 真机运行前的建议检查清单

正式跑真机前，建议逐项确认：

- UR5e IP 是否可达
- `/dev/ttyUSB0` 是否确实是 Inspire hand
- L515 是否能被 `pyrealsense2` 枚举
- Femto Bolt 是否能被 `pyorbbecsdk` 打开
- checkpoint 目录是否包含 `model.safetensors`、`metadata.pt`、`norm_stats.json`
- `train-config` 是否为 `pi05_pickplace_full_pytorch`
- 机械臂初始姿态是否在安全区
- 手爪初始开度是否合理
- 桌面和工件摆放是否和训练分布接近
- 降低 `--arm-speed` 与 `--arm-acceleration` 做首次验证

建议首次验证参数：

```bash
--control-hz 5 \
--arm-speed 0.03 \
--arm-acceleration 0.03
```

## 11. 我对当前系统的最终判断

### 已确认

- 训练完成模型确实已保存
- 最终 checkpoint 可直接指向 `.../60000`
- 代码中存在完整的最小真机推理闭环
- README 与脚本基本一致

### 尚未完全工程化

- 环境依赖仍有硬编码外部路径
- 缺少安全保护层
- 缺少设备自检脚本
- 缺少“只加载模型不驱动真机”的 dry-run 模式

## 12. 下一步建议

如果你要真正迁移到另一台电脑并稳定运行，我建议下一步优先做这三件事：

1. 把 `scripts/use_local_openpi_env.sh` 改造成“目标机可配置、无 RLinf-main 硬依赖”的版本。
2. 给 `scripts/pi05_real_robot_infer.py` 增加 `--dry-run`、设备自检和安全限幅。
3. 新增一个“一键真机前检查”脚本，先验证 camera / robot / hand / checkpoint / import，再决定是否进入 rollout loop。

