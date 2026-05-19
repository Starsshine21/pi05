# PI05 迁移方案（目标机没有 RLinf-main）

目标：把当前 `pi05/` 迁移到一台“什么都没有”的新电脑，并支持 **PI05 model 推理**，同时不再依赖 `/nfs_global/S/yangrongzheng/RLinf-main`。

## 结论先说

当前环境对 `RLinf-main` 的依赖，主要来自 `scripts/use_local_openpi_env.sh`：

- 借用 `RLinf-main/.venv` 里的 `site-packages`
- 借用 `RLinf_deps/libero`
- 借用其中若干 CUDA 相关动态库路径

这说明当前环境本质上是“拼接环境”，不是自包含环境。

**最稳的迁移办法不是复制 RLinf-main，而是把 `pi05` 变成一个独立 Python 环境。**

## 推荐迁移路线

推荐按下面三层拆开：

1. **代码层**：只迁移 `pi05/` 仓库代码。
2. **模型层**：单独拷贝你要推理用的 checkpoint，比如 `60000/`。
3. **环境层**：在目标机新建一个干净 conda/venv，把依赖直接装进这个环境，不再从 `RLinf-main` 借包。

## 你在目标机需要具备的东西

### 1. 基础系统

- Linux
- NVIDIA 驱动（如果要 GPU 推理）
- Python 3.11
- conda 或 mamba
- git

### 2. 真机推理额外依赖

如果要连真机，还需要：

- UR RTDE Python 包
- `pyserial`
- `pyrealsense2`
- `pyorbbecsdk`
- 对应硬件 SDK / udev / 驱动

## 最推荐的实际做法

## A. 先只做“模型可加载推理”

先别急着连真机。先在目标机做到：

- `openpi_official` 能 import
- checkpoint 能 load
- 能跑到 policy inference 初始化

这一步成功后，再补硬件 SDK。

## B. 新环境独立安装依赖

### 第一步：复制仓库

```bash
git clone <your pi05 repo>
cd pi05
```

### 第二步：创建干净环境

```bash
conda create -n pi05 python=3.11 -y
conda activate pi05
```

### 第三步：安装核心 Python 包

优先从 `openpi_official/pyproject.toml` 出发安装。

一个务实路线是：

```bash
cd openpi_official
pip install -e .
cd ..
pip install opencv-python pyserial
```

如果 GPU 推理需要严格匹配，再单独安装与你机器 CUDA 匹配的 `torch` / `jax`。

注意：这里的关键思想是：

- **所有包直接装进目标机这个新环境**
- **不要再通过 `PYTHONPATH` 指向 RLinf-main/.venv/site-packages**

## C. 改造环境脚本

当前 `scripts/use_local_openpi_env.sh` 依赖 RLinf-main，迁移时不要直接照搬原样。

目标机上建议改成只保留这几类内容：

- `PYTHONPATH=$REPO_ROOT/openpi_official/src`
- 必要的 cache 目录
- 可选的 `HF_HOME` / `OPENPI_DATA_HOME`

应该删除这类强依赖：

- `RLINF_ROOT`
- `RLINF_VENV_ROOT`
- `RLINF_VENV_SITE_PACKAGES`
- `LIBERO_REPO_PATH`

也就是说，迁移后的环境脚本应当是“自包含版本”。

## D. 模型文件如何带

你已经说模型可以自己传，所以代码仓库里不放 ckpt。

你到目标机后，只需要把例如下面这个目录单独拷过去：

```bash
results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000
```

推理最关键的是：

- `model.safetensors`
- `metadata.pt`
- `assets/local/pi05-pickplace-il/norm_stats.json`

## E. 真机推理分两步验

### 阶段 1：不连真机，只验模型侧

目标：确认 import 和 checkpoint 没问题。

### 阶段 2：逐个硬件验

按顺序验证：

1. 机器人 RTDE
2. Inspire hand 串口
3. L515
4. Femto Bolt

最后再跑整条真机链。

## 为什么我建议这样迁移

因为如果你把 `RLinf-main` 整坨复制过去：

- 体积大
- 路径脆弱
- 很多依赖不是 PI05 推理真正需要的
- 后续维护会越来越乱

而独立环境方式虽然第一次整理麻烦一点，但后面最省心。

## 你现在最该做的事

1. 先把当前 `pi05/` 代码推到 GitHub。
2. 目标机只拉代码，不拉 checkpoint。
3. 我再帮你把 `scripts/use_local_openpi_env.sh` 改成“无 RLinf-main 依赖版”。
4. 再给你补一个 `requirements` / `environment` 清单，专门用于目标机部署。

## 我建议的下一步

下一步最值的是直接做这两个改动：

1. 新增一个 `scripts/use_local_openpi_env_standalone.sh`
2. 新增一个“目标机部署说明”+“最小依赖清单”

这样你在另一台电脑上就不需要 RLinf-main 了。
