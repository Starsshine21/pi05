# PI05 / PI06 复现与真机推理完整说明

这份 `README.md` 是当前仓库的**主文档**，目标是把当前工作区里与 `pi05`、`pi06`、`openpi` 相关的内容整理成一份**从无到有可复现**的完整说明。

本文档重点覆盖：

1. 当前仓库里到底有什么。
2. 当前已经训练好的 PI05 模型保存在哪里。
3. 当前环境为什么依赖 `RLinf-main`。
4. 如何在当前机器复现数据、训练、checkpoint。
5. 如何把环境迁移到另一台“什么都没有”的电脑。
6. 如何在目标机完成 PI05 模型推理。
7. 当前真机链路是否完整，以及还缺什么。
8. PI06 / RECAP 相关内容在仓库中的位置。

---

## 1. 仓库定位

当前目录是一个以 `pi05` 为主、同时保留 `pi06` / RECAP 相关内容的工作区。

当前主线内容包括：

- `openpi_official/`：本地使用的官方 `openpi` 工作副本
- `openpi/`：另一个本地 openpi 工作副本
- `scripts/`：PI05 数据处理、训练、真机推理相关脚本
- `docs/`：PI05 训练与流程说明
- `pi05_sft/`：PI05 模型相关目录
- `pi06_recap/`：PI06 / RECAP 当前目录
- `archive/pi06_recap_legacy/`：旧版 PI06 RECAP 归档内容

当前仓库已经**不包含训练日志、数据、results checkpoint、models 大文件**，这些内容已经通过 `.gitignore` 忽略，代码仓只保留复现逻辑与文档。

这意味着：

- **代码、脚本、配置、文档可以完整复现**
- **训练数据、已训练 checkpoint、硬件 SDK 需要你在目标机另外准备**

所以这里说的“完整复现当前环境”，准确含义是：

- 可以完整复现当前的软件工程结构、训练/推理链路和环境搭建方式
- 但不能凭这个 Git 仓库单独恢复出被忽略掉的数据与模型文件


---

## 2. 当前已经训练好的 PI05 模型在哪里

根据训练日志 `logs/pi05-pickplace-pt-full-757027.err`，这次 full PyTorch 训练创建的实验目录是：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027
```

其中用于推理时最应该使用的最终 checkpoint 是：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

该目录下应至少包含：

- `model.safetensors`
- `metadata.pt`
- `optimizer.pt`
- `assets/local/pi05-pickplace-il/norm_stats.json`

真正推理时核心必需的是：

- `model.safetensors`
- `metadata.pt`
- `assets/local/pi05-pickplace-il/norm_stats.json`

如果你要在另一台机器推理，最稳妥的做法是**单独复制整个 `60000/` 目录**。

---

## 3. 当前环境为什么依赖 `RLinf-main`

当前仓库不是一个完全自包含环境，关键原因在：

```bash
scripts/use_local_openpi_env.sh
```

这个脚本目前会借用外部目录：

- `/nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages`
- `/nfs_global/S/yangrongzheng/RLinf-main/RLinf_deps/libero`
- 以及其中部分 CUDA 运行库路径

也就是说，当前环境是一个“**pi05 本地 conda 环境 + RLinf-main 里的 site-packages / 依赖补丁**”的拼接环境。

这也是为什么你说迁移到另一台什么都没有的机器时会头疼：

- 当前机器能跑，不代表环境本身独立完整
- 直接复制仓库并不能保证目标机能 import 成功

不过这不是无解，后面会给出两种完整复现路线：

1. **兼容当前环境的 RLinf-main 路线**：最贴近现状，风险小
2. **去 RLinf-main 的独立环境路线**：更干净，更适合长期迁移

---

## 4. 仓库里的关键目录说明

### 4.1 PI05 主线相关

- `scripts/convert_pick_place_to_lerobot.py`：把原始 pick-place 数据转成 LeRobot 格式
- `scripts/pi05_real_robot_infer.py`：PI05 真机推理脚本
- `scripts/pi05_official_pickplace_pytorch_full.slurm`：PI05 官方 full PyTorch 训练脚本
- `scripts/pi05_official_pickplace_pytorch_full_smoke.slurm`：smoke 训练脚本
- `scripts/use_local_openpi_env.sh`：当前环境注入脚本
- `openpi_official/`：官方 openpi 工作副本

### 4.2 PI06 / RECAP 相关

- `pi06_recap/`：当前 PI06 / RECAP 目录
- `archive/pi06_recap_legacy/`：历史归档版 RECAP 内容
- `recap_workspace/`：与 RECAP 相关的工作目录

如果你当前重点是 **PI05 model 推理与真机部署**，优先关注：

- `scripts/`
- `openpi_official/`
- `README.md`

---

## 5. 当前 PI05 训练任务的输入输出定义

当前已经接到官方 `openpi` 路线上的 PI05 任务，本质是：

- 文本 + 图像 + 状态 -> 动作

### 输入 observation

当前训练/推理使用的 observation 语义是：

- `prompt`
- `image`
- `wrist_image`
- `state`

其中：

- `image`：来自 `L515`
- `wrist_image`：来自 `Orbbec Femto Bolt`
- `state`：`concat([joints, eef, hand])`，总共 18 维

### 输出 action

当前真机脚本把模型输出解释为前 12 维：

- 前 6 维：`joint delta`
- 后 6 维：`hand delta`

即：

```python
target_joints = current_joints + action[:6]
target_hand = current_hand + action[6:12]
```

然后：

- 通过 UR5e RTDE 下发关节命令
- 通过 Inspire hand serial 下发手爪命令

---

## 6. 从零开始复现：总览

如果你要“从无到有完整复现”，可以分成 6 个阶段：

1. 准备代码仓库
2. 准备 Python / CUDA 环境
3. 准备 LeRobot 数据
4. 计算 norm stats
5. 启动 PI05 训练或加载已有 checkpoint
6. 做本地推理 / 真机推理

下面分别给出两条路线：

- **路线 A：继续依赖 RLinf-main（最贴近当前环境）**
- **路线 B：完全独立重建环境（更推荐长期迁移）**

---

## 7. 路线 A：继续依赖 `RLinf-main` 的完整复现

这条路线最适合：

- 你想尽快在另一台机器上复现“和当前机器最接近”的行为
- 你不介意目标机上也准备一个 `RLinf-main`
- 你更看重成功率，而不是环境整洁度

### 7.1 目标机准备基础环境

目标机建议具备：

- Linux
- NVIDIA 驱动
- CUDA 兼容环境
- Python 3.11
- conda / mamba
- git

### 7.2 拉代码

目标机上执行：

```bash
git clone https://github.com/Starsshine21/pi05.git
cd pi05
```

同时准备一份 `RLinf-main`，路径建议仍保持：

```bash
/nfs_global/S/yangrongzheng/RLinf-main
```

如果路径不同，也可以，后面要改环境变量。

### 7.3 准备 conda 环境

如果你要尽量贴近当前机器，建议建立一个同名环境目录：

```bash
conda create -p /path/to/pi05/.conda-pi05-openpi-final python=3.11 -y
conda activate /path/to/pi05/.conda-pi05-openpi-final
```

然后把当前运行 PI05 所需的包安装进去。

### 7.4 让环境脚本可用

当前环境脚本是：

```bash
scripts/use_local_openpi_env.sh
```

你需要确保其中这些路径在目标机上是存在的：

- `PI05_CONDA_ENV_ROOT`
- `RLINF_ROOT`
- `RLINF_VENV_ROOT`
- `RLINF_VENV_SITE_PACKAGES`
- `LIBERO_REPO_PATH`

如果你的目标机路径不同，可以用环境变量覆盖：

```bash
export PI05_CONDA_ENV_ROOT=/path/to/pi05/.conda-pi05-openpi-final
export RLINF_ROOT=/path/to/RLinf-main
export RLINF_VENV_ROOT=/path/to/RLinf-main/.venv
export RLINF_VENV_SITE_PACKAGES=/path/to/RLinf-main/.venv/lib/python3.11/site-packages
export LIBERO_REPO_PATH=/path/to/RLinf-main/RLinf_deps/libero
source scripts/use_local_openpi_env.sh
```

### 7.5 准备 LeRobot 数据

当前本地转换后的数据目录原本是：

```bash
data/lerobot_pick_place
```

如果你还没有数据，需要先把原始数据转成 LeRobot：

```bash
python scripts/convert_pick_place_to_lerobot.py --overwrite --image-height 224 --image-width 224
```

### 7.6 建立 LeRobot 软链

当前官方 loader 使用的 repo id 是：

```bash
local/pi05-pickplace-il
```

因此要保证目标机存在类似软链：

```bash
mkdir -p /path/to/huggingface/lerobot/local
ln -s /path/to/pi05/data/lerobot_pick_place /path/to/huggingface/lerobot/local/pi05-pickplace-il
```

如果沿用当前脚本默认行为，一般会链接到：

```bash
$RLINF_ROOT/models/huggingface/lerobot/local/pi05-pickplace-il
```

### 7.7 计算 norm stats

进入 `openpi_official`：

```bash
cd openpi_official
```

执行：

```bash
PYTHONPATH=/path/to/pi05/openpi_official/src:$PYTHONPATH \
python scripts/compute_norm_stats.py --config-name pi05_pickplace_lora
```

对于 full PyTorch 配置，训练日志表明当前 full 训练复用了已有 norm stats，没有在 full train 前重新计算。

### 7.8 启动训练

当前 full PyTorch 训练脚本是：

```bash
scripts/pi05_official_pickplace_pytorch_full.slurm
```

其中关键配置包括：

- `TRAIN_CONFIG_NAME=pi05_pickplace_full_pytorch`
- `NORM_CONFIG_NAME=pi05_pickplace_lora`
- `--checkpoint-base-dir results/openpi_official_pytorch_full_checkpoints`

如果你是在集群上，可直接提交该 slurm。

如果你是本地单机，要参考这个 slurm 脚本，把训练命令改写成单机可运行形式。

### 7.9 使用已有 checkpoint 推理

如果你不想重训，只做推理，则只需要把 checkpoint 目录单独传到目标机，比如：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

---

## 8. 路线 B：不依赖 `RLinf-main` 的完整复现

这条路线更适合：

- 目标机什么都没有
- 你不想再维护 `RLinf-main`
- 你希望最终环境干净且更容易迁移

这条路线的原则是：

- **所有依赖都直接装进目标机自己的 Python 环境**
- **不再从 `RLinf-main/.venv/site-packages` 借包**
- **只保留当前仓库 + checkpoint + 硬件 SDK**

### 8.1 拉代码

```bash
git clone https://github.com/Starsshine21/pi05.git
cd pi05
```

### 8.2 创建独立环境

```bash
conda create -n pi05 python=3.11 -y
conda activate pi05
```

### 8.3 安装 openpi 官方依赖

先安装 `openpi_official/`：

```bash
cd openpi_official
pip install -e .
cd ..
```

然后补充当前仓库里单独列出的最小补充包：

```bash
pip install -r requirements-standalone.txt
```

注意这里有一个现实问题：`openpi_official/pyproject.toml` 中默认依赖包含：

- `torch==2.7.1`
- `jax[cuda12]==0.5.3`
- `lerobot`

因此独立路线能否一次安装成功，取决于目标机的：

- CUDA 驱动版本
- Python 3.11 兼容性
- 网络与 pip/uv 源可用性

如果你**只做推理**，优先目标是先让这些包装成功；如果你**还要训练**，则还要保证 `jax[cuda12]`、`lerobot` 等训练相关依赖可用。

当前 `openpi_official/pyproject.toml` 可以作为主要安装依据，但它不是“任何机器零摩擦必装成功”的保证。

### 8.4 使用 standalone 环境脚本

本仓库已经新增了一个**不影响当前本地工作流**的新脚本：

```bash
scripts/use_local_openpi_env_standalone.sh
```

它和原有的：

```bash
scripts/use_local_openpi_env.sh
```

是并行存在的。原脚本继续服务你当前机器；新脚本只服务目标机独立复现。

目标机上使用方式：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate pi05
source scripts/use_local_openpi_env_standalone.sh
```

这个脚本会自动：

- 设置 `PYTHONPATH` 指向 `openpi_official/src`
- 设置 `HF_HOME` / `HF_DATASETS_CACHE` / `OPENPI_DATA_HOME`
- 如果本地存在 `data/lerobot_pick_place`，自动在 `.cache/lerobot/local/pi05-pickplace-il` 建软链

注意：该软链只是在仓库本地准备一个 `local/pi05-pickplace-il` 风格的数据入口。是否能被 `lerobot` / `openpi` 正确读取，仍取决于目标机上安装的 `lerobot` 行为与缓存路径。也就是说，这一步是“兼容准备”，不是 100% 替代所有 Hugging Face / LeRobot 默认路径逻辑。

### 8.5 用检查脚本验证环境

新增检查脚本：

```bash
scripts/check_standalone_openpi_env.py
```

使用方式：

```bash
python scripts/check_standalone_openpi_env.py
```

它会检查：

- `cv2`
- `numpy`
- `torch`
- `serial`
- `openpi.training.config`
- `openpi.policies.policy_config`

并额外尝试检查硬件依赖：

- `pyrealsense2`
- `pyorbbecsdk`
- `rtde_control`
- `rtde_receive`

### 8.6 安装真机相关依赖

如果要做真机推理，还需要准备：

- `rtde_control`
- `rtde_receive`
- `pyrealsense2`
- `pyorbbecsdk`
- `pyserial`

其中：

- `pyrealsense2` 依赖 Intel RealSense SDK
- `pyorbbecsdk` 依赖 Orbbec SDK
- UR RTDE 依赖 UR 官方 Python RTDE 包或兼容安装方式

### 8.7 准备数据 / norm stats / checkpoint

如果要训练，仍然需要：

1. 原始数据
2. LeRobot 数据转换
3. `local/pi05-pickplace-il` 链接
4. norm stats

如果只做推理，则最关键的是：

- `openpi_official/src`
- 可用的 Python 环境
- checkpoint `60000/`
- 对应的 `norm_stats.json`

如果只做**离线 checkpoint 加载 + policy 初始化**，你可以不准备原始训练数据；但如果你要重训、重算 norm stats，或者验证完整训练链，就必须另外准备原始数据或转换后的 `data/lerobot_pick_place`。

## 9. 数据复现：原始数据到 LeRobot

当前仓库的数据转换主脚本是：

```bash
scripts/convert_pick_place_to_lerobot.py
```

用途：

- 把真机原始 pick-place 数据转换成 LeRobot 数据集
- 输出目录是：

```bash
data/lerobot_pick_place
```

典型执行方式：

```bash
python scripts/convert_pick_place_to_lerobot.py --overwrite --image-height 224 --image-width 224
```

转换完成后，再通过软链暴露给官方 loader：

```bash
mkdir -p <HF_LEROBOT_ROOT>/local
ln -s /path/to/pi05/data/lerobot_pick_place <HF_LEROBOT_ROOT>/local/pi05-pickplace-il
```

---

## 10. 训练复现：norm stats 与 full PyTorch 训练

### 10.1 norm stats

如果你的资产目录还没有统计文件，可以在 `openpi_official/` 下执行：

```bash
PYTHONPATH=/path/to/pi05/openpi_official/src:$PYTHONPATH \
python scripts/compute_norm_stats.py --config-name pi05_pickplace_lora
```

### 10.2 full PyTorch 训练

当前 full 训练的标准 slurm 脚本是：

```bash
scripts/pi05_official_pickplace_pytorch_full.slurm
```

脚本里关键点包括：

- conda 环境：`.conda-pi05-openpi-final`
- `source scripts/use_local_openpi_env.sh`
- 训练配置：`pi05_pickplace_full_pytorch`
- checkpoint 根目录：`results/openpi_official_pytorch_full_checkpoints`
- assets 根目录：`openpi_official/assets`

当前 `757027` 这次 full 训练的实验目录是：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027
```

最终 checkpoint 为：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

---

## 11. 在另一台机器上只做 PI05 模型推理

如果你的目标只是“把当前环境迁到另一台电脑上跑 PI05 model 推理”，最小必需项是：

### 11.1 代码

- 当前 GitHub 仓库代码
- 至少要有 `openpi_official/`
- 至少要有 `scripts/pi05_real_robot_infer.py`

### 11.2 模型

单独复制：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

### 11.3 Python 环境

确保能 import：

- `openpi.training.config`
- `openpi.policies.policy_config`
- `torch`
- `cv2`
- `numpy`

### 11.4 真机依赖（如果要上真机）

- `rtde_control`
- `rtde_receive`
- `serial`
- `pyrealsense2`
- `pyorbbecsdk`

---

## 12. 真机推理：当前链路是否已经完整

结论先说：

- **研究验证级别：基本完整**
- **工程部署级别：还不完整**

### 12.1 已经具备的部分

当前 `scripts/pi05_real_robot_infer.py` 已经实现：

- 加载 `openpi_official` 训练得到的 checkpoint
- 采集 `L515` 图像
- 采集 `Orbbec Femto Bolt` 图像
- 读取 UR5e 当前关节 / TCP
- 读取 Inspire hand 当前状态
- 组织 observation：`image / wrist_image / state / prompt`
- 调用 policy inference
- 把输出动作解释成 `joint delta + hand delta`
- 下发到机械臂和手爪

从“最小闭环”角度看，真机主链路已经打通。

### 12.2 当前还缺什么

当前还缺这些工程化能力：

- 关节限位保护
- 工作空间边界保护
- 自碰撞 / 桌面碰撞保护
- 速度和加速度二次限幅
- dry-run 模式
- 设备自检脚本
- checkpoint / config 匹配检查
- 首帧人工确认 / 安全门控

所以现在的状态更适合：

- 小范围人工盯守验证
- 研究实验
- 先跑通再逐步加安全层

而不是直接无保护上线。

---

## 13. 真机推理命令

当前最小可用真机推理命令如下：

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

如果你是在另一台机器上运行，需要把路径改成目标机实际路径。

---

## 14. 真机推理前建议检查清单

建议在上真机前逐项确认：

- UR5e IP 可以 ping 通
- `/dev/ttyUSB0` 确实对应 Inspire hand
- `pyrealsense2` 能打开 L515
- `pyorbbecsdk` 能打开 Femto Bolt
- checkpoint 目录包含 `model.safetensors`、`metadata.pt`、`norm_stats.json`
- `train-config` 使用的是 `pi05_pickplace_full_pytorch`
- 机械臂当前位置在安全区域
- 桌面摆放与训练分布相近
- 首次测试时降低速度和加速度

建议首次测试使用更保守参数：

```bash
--control-hz 5 \
--arm-speed 0.03 \
--arm-acceleration 0.03
```

---

## 15. 另一台“什么都没有”的电脑，推荐怎么做

如果你的目标机现在什么都没有，我建议按这个优先级：

### 方案 1：先求能跑

- 拉当前 GitHub 仓库
- 单独传 `60000/` checkpoint
- 在目标机安装独立 Python 环境
- 把 `openpi_official` 依赖装进去
- 先做到“能 import、能 load checkpoint、能本地推理初始化”
- 最后再逐个补机器人 / 相机 / 手爪 SDK

这是最实用、最稳的方式。

### 方案 2：完全复刻当前机

- 目标机同时准备 `RLinf-main`
- 尽量保留与当前机一致的路径布局
- 使用 `scripts/use_local_openpi_env.sh`
- 补齐 `.conda-pi05-openpi-final` 等环境

这是最贴近当前机行为的方式，但维护成本更高。

如果你是第一次迁移，我更推荐**方案 1**。

---

## 16. PI06 / RECAP 内容如何看

当前仓库除了 PI05，还保留了 PI06 / RECAP 相关内容：

- 当前目录：`pi06_recap/`
- 历史归档：`archive/pi06_recap_legacy/`
- 相关工作目录：`recap_workspace/`

如果你之后要继续做 PI06 / RECAP，可以在这些目录继续展开；但当前这份 README 的主线重点仍然是：

- PI05 数据到 LeRobot
- PI05 openpi 训练
- PI05 checkpoint 推理
- PI05 真机推理链

---

## 17. 当前结论总结

### 已经确认的事实

- 当前有训练完成的 PI05 full PyTorch checkpoint
- 最终 checkpoint 是：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

- 当前仓库已经有最小真机推理脚本：`scripts/pi05_real_robot_infer.py`
- 当前真机链路在“研究验证”意义上已经基本完整
- 当前环境确实依赖 `RLinf-main`
- 但完全可以通过目标机独立重建环境来摆脱这种依赖
- 不过“摆脱 `RLinf-main`”不等于“什么都不用装”，目标机仍然必须补齐 `openpi_official`、`lerobot`、`torch/jax`、以及真机 SDK 依赖

### 如果你现在就要落地迁移

最推荐顺序是：

1. 目标机拉代码仓
2. 单独传 `60000/` 模型目录
3. 建一个新的 Python 3.11 环境
4. 安装 `openpi_official` 依赖
5. 先验证 checkpoint load
6. 再逐个验证 L515 / Orbbec / UR5e / Inspire hand
7. 最后再开真机闭环

---

## 18. 本仓库当前推荐阅读顺序

如果你第一次接手这个仓库，建议按这个顺序读：

1. `README.md`
2. `scripts/use_local_openpi_env.sh`
3. `scripts/convert_pick_place_to_lerobot.py`
4. `scripts/pi05_official_pickplace_pytorch_full.slurm`
5. `scripts/pi05_real_robot_infer.py`
6. `openpi_official/src/openpi/training/config.py`

这样能最快建立从“环境 -> 数据 -> 训练 -> 推理 -> 真机”的完整认识。

---

## 9. 真机部署与 ReCap 在线数据闭环指南

这一节总结当前仓库里已经补齐的 **PI05 真机部署 → 在线采集 → 转 LeRobot → 生成 returns → 接入 ReCap value training** 的整体流程。

### 9.1 当前已经实现了什么

当前仓库已经具备下面这些能力：

- 使用训练好的 `pi05` checkpoint 在真机上做在线推理
- 通过真机 API 读取：
  - 机械臂关节
  - 末端位姿
  - Inspire hand 状态
  - 两路相机图像
- 将 observation 组织成当前 `pi05` / `ReCap` 对齐的数据契约：
  - `image`
  - `wrist_image`
  - `state`
  - `prompt`
- 将 rollout 记录成可转训练集的原始缓存
- 将 rollout 转成和现有 `data/lerobot_pick_place` 一致的 LeRobot 数据格式
- 为在线数据生成 sparse terminal reward 的 returns sidecar
- 使用 offline + online 数据混合继续训练 ReCap value model

也就是说，目前已经实现了：

**真机 rollout → 在线数据落盘 → LeRobot 转换 → sparse returns → mixed value training**

但还**没有**实现完整的在线 actor-critic / 真机 RL 策略更新闭环。

---

### 9.2 真机部署的核心脚本

#### 1. 真机推理脚本

- `scripts/pi05_real_robot_infer.py`

这份脚本已经实现：

- UR5e RTDE 控制
- Inspire hand 串口控制
- L515 与 Orbbec 相机读取
- checkpoint 加载
- `policy.infer(obs)` 在线推理
- action 下发到真机

当前 observation 组织方式：

- `image`
- `wrist_image`
- `state`
- `prompt`

当前 action 解释方式：

- 前 6 维：arm joint delta
- 后 6 维：hand delta

这与当前 `pi05` / `ReCap` 的数据契约已经对齐。

---

#### 2. 真机 ReCap 采集脚本

- `scripts/pi05_recap_real_collect.py`

这是在 `pi05_real_robot_infer.py` 基础上补出来的 rollout recorder。

功能：

- 真机跑当前 `pi05` policy
- 每一步记录：
  - `image`
  - `wrist_image`
  - `state`
  - `actions`
  - `prompt`
  - `timestamp`
- episode 结束后人工确认 success / failure
- 保存到：
  - `raw_rollouts/episode_xxxxxx/frames.npz`
  - `raw_rollouts/episode_xxxxxx/meta.json`

这一步先不直接写 LeRobot，而是先保存成稳定的原始缓存格式，便于回放和调试。

---

#### 3. 原始 rollout 转 LeRobot

- `scripts/convert_pi05_rollout_to_lerobot.py`

功能：

- 读取 `raw_rollouts/episode_xxxxxx`
- 转成和现有训练数据一致的 LeRobot 本地数据集格式
- 输出目录默认：
  - `/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_online`

输出结构会与现有离线数据风格一致，例如：

- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `data/chunk-000/episode_xxxxxx.parquet`

---

#### 4. 生成 sparse returns

- `scripts/generate_sparse_returns.py`

功能：

- 为在线 LeRobot 数据集生成 sparse terminal reward sidecar
- 输出例如：
  - `meta/returns_local_pi05_online.parquet`

当前 reward 规则：

- 成功轨迹：最后一步 `reward = 1`
- 其它步：`reward = 0`
- 当前默认 `return = 1`（成功轨迹）

这与当前 ReCap value training 的 sparse reward 设定一致。

---

### 9.3 真实部署 / 采集的标准流程

#### 步骤 1：运行真机 rollout 采集

示例：

```bash
python scripts/pi05_recap_real_collect.py \
  --checkpoint-dir /path/to/pi05/checkpoint \
  --train-config pi05_pickplace_full_pytorch \
  --prompt "place the apple at the left center position on the puzzle" \
  --output-dir /nfs_global/S/yangrongzheng/pi05/data/recap_real_collect \
  --num-episodes 1
```

说明：

- 每个 episode 结束后会让你手工确认是否 success
- 采样频率由 `--control-hz` 控制，默认 `10Hz`

---

#### 步骤 2：转换成 LeRobot 在线数据集

```bash
python scripts/convert_pi05_rollout_to_lerobot.py \
  --input-dir /nfs_global/S/yangrongzheng/pi05/data/recap_real_collect \
  --output-dir /nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_online \
  --overwrite
```

---

#### 步骤 3：生成 sparse returns sidecar

```bash
python scripts/generate_sparse_returns.py \
  --dataset-dir /nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_online \
  --tag local_pi05_online
```

生成结果：

- `/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_online/meta/returns_local_pi05_online.parquet`

---

### 9.4 ReCap 如何接入 online 数据

为了支持 offline + online 混合训练，我已经补了一份配置：

- `recap_workspace/configs/local_value_sft_online.yaml`

以及对应训练脚本：

- `recap_workspace/scripts/recap_value_train_online_1gpu.slurm`

这份 mixed config 会同时吃：

- 离线 demonstration 数据：
  - `/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place`
- 真机新采 online 数据：
  - `/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_online`

也就是说，当前已经支持：

**offline value data + online rollout data → mixed value training**

启动方式：

```bash
sbatch recap_workspace/scripts/recap_value_train_online_1gpu.slurm
```

---

### 9.5 当前 ReCap 在线流程到哪一步了

当前已经打通：

1. 真机跑 `pi05`
2. 记录 rollout
3. 转 LeRobot
4. 生成 sparse returns
5. 接入 mixed value training

所以目前已经具备：

**在线数据进入 ReCap value model 训练** 的第一版能力。

---

### 9.6 当前还没有实现的部分

虽然现在已经具备在线数据闭环的第一版，但还没有实现完整“真机 RL / 策略迭代”系统。

还缺的主要部分包括：

- 自动将 online 数据 append / merge 到历史数据集
- 自动 resume value training
- 基于 value 的 policy rerank / filtering / improvement
- 真机 rollout 的安全控制增强：
  - action clipping
  - emergency stop
  - reset 管理
  - 自动失败检测
- 真正意义上的 actor-critic / online RL 策略更新闭环

所以当前状态更准确地说是：

- **离线 + 在线 value data pipeline 已打通**
- **完整真机 RL policy improvement 还没有完全补齐**

---

### 9.7 推荐的使用策略

当前建议按以下顺序使用：

#### 第一阶段：继续训练 offline value
- 用当前离线 demonstration 数据把 value model 训稳

#### 第二阶段：开始小规模真机 rollout
- 用 `pi05_recap_real_collect.py` 收少量真实 rollout
- 转成 online LeRobot 数据
- 生成 sparse returns

#### 第三阶段：做 mixed value training
- 使用 `local_value_sft_online.yaml`
- 同时训练 offline + online 数据

#### 第四阶段：再往 policy improvement 演化
- 后续再补：
  - rerank
  - filtering
  - offline-to-online improvement
  - 真正在线 RL 更新

---

### 9.8 相关文件索引

真机与 ReCap 在线流程相关的关键文件如下：

- 真机推理：
  - `scripts/pi05_real_robot_infer.py`
- 真机 rollout 采集：
  - `scripts/pi05_recap_real_collect.py`
- rollout 转 LeRobot：
  - `scripts/convert_pi05_rollout_to_lerobot.py`
- sparse returns 生成：
  - `scripts/generate_sparse_returns.py`
- offline value 训练配置：
  - `recap_workspace/configs/local_value_sft.yaml`
- offline+online mixed value 配置：
  - `recap_workspace/configs/local_value_sft_online.yaml`
- offline value 训练脚本：
  - `recap_workspace/scripts/recap_value_train_1gpu.slurm`
- mixed online value 训练脚本：
  - `recap_workspace/scripts/recap_value_train_online_1gpu.slurm`

