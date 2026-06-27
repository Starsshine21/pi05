# RoboTwin 下跑通 PI05：环境复现与使用说明

本文档记录当前仓库里，如何把 `external/RoboTwin/policy/pi05` 这条链路真正跑起来，包括：

- 环境怎么复现
- 关键路径怎么配置
- 当前已经修掉的兼容问题
- 如何做最小验证
- 后续如何运行 RoboTwin 的 `pi05` policy
- 出问题时优先看哪里

本文档针对当前机器上的这套可用状态整理，目标是“换一台机器后，照着做能尽量复现出来”。

---

## 1. 当前已经跑通到哪一步

当前真实状态：

- RoboTwin 环境代码可导入
- `policy/pi05` 的 OpenPI 代码链可导入
- RoboTwin 本地 `pi05` checkpoint 可识别
- Hugging Face 分片 `safetensors` checkpoint 可加载
- RoboTwin 的 `get_model(...)` 已成功返回 `PI0` 模型实例

已经实测成功的最小链路：

```bash
cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/external/RoboTwin:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/src:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/packages/openpi-client/src:$PYTHONPATH
export MPLCONFIGDIR=/tmp

python - <<'PY'
from policy.pi05.deploy_policy import get_model
usr_args = {
    "train_config_name": "pi05_aloha_full_base",
    "model_name": "model_robotwin",
    "checkpoint_id": 30000,
    "pi0_step": 50,
}
model = get_model(usr_args)
print("OK: model loaded", type(model).__name__)
PY
```

成功标志：

- 日志里出现 `loading model success!`
- 最终打印 `OK: model loaded PI0`

注意：这说明**模型构造链路已经打通**，但不等于完整 RoboTwin rollout 已经完成。完整 rollout 还要继续验证任务环境与仿真执行。

补充进展（2026-06-04 / 2026-06-05）：

- `scripts/robotwin_pi05_adjust_bottle_smoke.slurm` 已真实提交并跑起（job `763500`）
- `adjust_bottle` rollout 已进入真实仿真执行，而不只是停留在 `get_model(...)`
- 日志 `logs/rt-pi05-adj1-763500.out` 中可见：
  - `loading model success!`
  - 多条 `Episode XXX | SUCCESS`
  - 多条 `[adjust_bottle] success=True`
- 新结果目录已落盘：

```bash
external/RoboTwin/eval_result/adjust_bottle/pi05/demo_clean/model_robotwin/2026-06-04 21:29:04
```

这说明 **RoboTwin + pi05 policy 的推理 / rollout 链路已经跑通**。

但这仍然**不等于 value model 训练已经跑通**；value model 属于 `recap_workspace` 下的另一条训练链路，见后文补充说明。

---

## 2. 本次修过的关键兼容问题

这次真正修掉了两类问题。

### 2.1 HF 分片 safetensors 加载问题

RoboTwin 的 `pi05` checkpoint 不是单文件：

- `model-00001-of-00003.safetensors`
- `model-00002-of-00003.safetensors`
- `model-00003-of-00003.safetensors`
- `model.safetensors.index.json`

原始 loader 会把 `model.safetensors.index.json` 当成真正的权重文件读，导致报错：

- `safetensors_rust.SafetensorError: Error while deserializing header: header too large`

已经修复为：

- 识别 `model.safetensors.index.json`
- 读取 `weight_map`
- 顺序加载所有 shard
- 合并成 `state_dict`
- 再喂给 PyTorch 模型

对应修改文件：

- `external/RoboTwin/policy/pi05/src/openpi/models/model.py`

### 2.2 checkpoint 额外 value head 导致加载失败

当前 `model_robotwin` checkpoint 里带有训练期额外头：

- `value_head.mlp.*`

但推理模型 `PI0Pytorch` 并没有这个模块。

原来在 shard 合并后会报：

- `Unexpected keys while loading ... value_head.mlp.*`

已经修复为：

- 对 `value_head.*` 这类 checkpoint-only keys 只 warning，不再 hard fail

这符合当前推理场景，因为这些权重不是 policy 前向所需主干。

### 2.3 frozen dataclass 赋值问题

RoboTwin 的 `policy_config.py` 里会在运行时写：

- `data_config.asset_id = robotwin_repo_id`

但 `DataConfig` 是 `frozen=True` 的 dataclass，因此会报：

- `FrozenInstanceError: cannot assign to field 'asset_id'`

已经修复为：

- 使用 `dataclasses.replace(data_config, asset_id=robotwin_repo_id)`

对应修改文件：

- `external/RoboTwin/policy/pi05/src/openpi/policies/policy_config.py`

### 2.4 RECAP / value model 训练链路的当前状态

`recap_workspace` 下的 value model 训练，和 RoboTwin policy rollout 不是同一条链：

- RoboTwin rollout 跑的是 policy 推理 / eval
- `recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm` 跑的是 value model 训练

当前对 value model 训练链的真实结论是：

- 旧日志 `recap_workspace/logs/recap-adj100-763373.err`
- 旧日志 `recap_workspace/logs/recap-adj100-763379.err`

都表明训练曾卡在：

- `TypeError: expected bytes, PosixPath found`

为此，本仓库已在：

- `recap_workspace/run_value_local.py`

里增加本地 patch，用于：

- 清理 `sys.path`，优先使用仓库内 vendored `rlinf/openpi/libero`
- patch `ray.init(...)` 的 path-like 参数
- patch `ray._raylet.CoreWorker(...)` 的 path-like 参数
- patch若干 `lerobot` 兼容行为

截至本文档本次更新（2026-06-05），**还没有拿到一份新的、能证明 value model 已经成功进入训练 step 的真实日志**。

当前新暴露的阻塞点不是旧的 `PosixPath found`，而是：

- 在当前交互环境里直接本地启动 Ray head 时，命中
  - `PermissionError: [Errno 1] Operation not permitted`
  - 位置在 `ray._private.services.get_node_ip_address()` 创建 socket 时

这说明：

- 旧的 `PosixPath found` 根因已经被进一步绕开/推进到更后面的阶段
- 但 **value model 训练是否真正跑通，仍需一份新的真实训练日志来最终确认**

因此当前应把状态明确区分为：

- **RoboTwin policy rollout：已跑通**
- **RECAP value model 训练：仍在推进中，尚未拿到最终成功证据**

---

## 3. 当前可用环境

当前确认可用的是这个 conda 环境：

```bash
/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
```

激活方式：

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
```

这个环境中已确认：

- `numpy==1.26.4`
- `torch==2.6.0+cu124`

如果默认 `python` 不在这个环境里，你直接运行 RoboTwin 会遇到类似：

- `ModuleNotFoundError: No module named 'numpy'`

所以后续所有命令都建议先显式激活这个 conda 环境。

---

## 4. 一台新机器上怎么复现环境

如果你不是直接复用当前 `.conda-pi05-openpi-final`，建议按下面顺序复现。

### 4.1 准备 Python 环境

推荐 Python 3.11：

```bash
conda create -n pi05 python=3.11 -y
conda activate pi05
```

### 4.2 安装 RoboTwin pi05 依赖

进入：

```bash
cd /path/to/pi05/external/RoboTwin/policy/pi05
```

优先按它自己的 `pyproject.toml` 安装：

```bash
pip install -e .
```

如果有额外仓库级依赖，也可以补装当前主仓库的依赖：

```bash
cd /path/to/pi05
pip install -r requirements-standalone.txt
```

### 4.3 确保 PyTorch / CUDA 版本匹配

当前跑通环境里是：

- `torch 2.6.0+cu124`

如果你换机器复现，至少要保证：

- `torch` 可用
- GPU 可见时，CUDA 版本与本地驱动兼容
- `transformers` / `safetensors` / `flax` / `jax` 能同时 import

### 4.4 最低限度导入验证

在新环境中，先跑：

```bash
python - <<'PY'
import numpy
import torch
import safetensors
import jax
import flax
print('basic imports ok')
PY
```

通过后，再继续 RoboTwin/OpenPI 链路验证。

---

## 5. 目录和路径要求

当前 RoboTwin `pi05` 默认依赖这些路径关系。

### 5.1 代码目录

仓库根目录：

```bash
/nfs_global/S/yangrongzheng/pi05
```

RoboTwin 目录：

```bash
/nfs_global/S/yangrongzheng/pi05/external/RoboTwin
```

RoboTwin 的 pi05 代码目录：

```bash
/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05
```

### 5.2 checkpoint 目录

RoboTwin 当前有这两个模型目录：

```bash
external/RoboTwin/policy/pi05/checkpoints/pi05_aloha_robotwin/model_robotwin
external/RoboTwin/policy/pi05/checkpoints/pi05_aloha_full_base/model_robotwin
```

当前测试使用的是：

- `train_config_name = pi05_aloha_full_base`
- `model_name = model_robotwin`
- `checkpoint_id = 30000`

而 `pi_model.py` 内部最终会去找：

```bash
policy/pi05/checkpoints/<train_config_name>/<model_name>/<checkpoint_id>
```

也就是：

```bash
external/RoboTwin/policy/pi05/checkpoints/pi05_aloha_full_base/model_robotwin/30000
```

同时它还要求这个目录下有：

```bash
assets/
```

并从 `assets/` 里自动取第一个子目录作为 `robotwin_repo_id`。

### 5.3 tokenizer 路径

当前成功加载时，会打印本机 tokenizer 路径：

```bash
/nfs_global/S/yangrongzheng/RLinf-main/models/openpi/big_vision/paligemma_tokenizer.model
```

如果你换机器复现，这个路径可能不存在。那就需要检查：

- `openpi` 的 tokenizer 查找逻辑
- 相关模型资源是否已下载到本地

如果 tokenizer 缺失，模型加载会在更早阶段失败。

---

## 6. 运行前必须设置的环境变量

当前最稳妥的方式是显式设置 `PYTHONPATH`。

在 `external/RoboTwin` 下运行前，建议：

```bash
export PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/external/RoboTwin:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/src:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/packages/openpi-client/src:$PYTHONPATH
export MPLCONFIGDIR=/tmp
```

说明：

- `external/RoboTwin`：保证 `policy.pi05...`、`envs...` 这些包可导入
- `policy/pi05/src`：保证 `openpi...` 可导入
- `policy/pi05/packages/openpi-client/src`：保证 client 相关包可导入
- `MPLCONFIGDIR=/tmp`：避免 matplotlib 在不可写 home 目录下报 cache warning

如果 `/tmp` 不可写，可以改成你自己的目录，例如：

```bash
mkdir -p /path/to/mplcache
export MPLCONFIGDIR=/path/to/mplcache
```

---

## 7. 推荐的最小验证顺序

建议按从轻到重的顺序验证，不要一上来就跑完整仿真。

### 7.1 第一步：验证基础依赖

```bash
python - <<'PY'
import numpy, torch, jax, flax, safetensors
print('deps ok')
PY
```

### 7.2 第二步：验证 OpenPI / RoboTwin import

```bash
cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin

python - <<'PY'
import openpi
import policy.pi05.deploy_policy
print('imports ok')
PY
```

### 7.3 第三步：验证 PI0 直接初始化

```bash
cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin

python - <<'PY'
from policy.pi05.pi_model import PI0
model = PI0('pi05_aloha_full_base', 'model_robotwin', 30000, 50)
print('PI0 init ok')
PY
```

### 7.4 第四步：验证 RoboTwin 的 get_model 入口

```bash
cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin

python - <<'PY'
from policy.pi05.deploy_policy import get_model
usr_args = {
    'train_config_name': 'pi05_aloha_full_base',
    'model_name': 'model_robotwin',
    'checkpoint_id': 30000,
    'pi0_step': 50,
}
model = get_model(usr_args)
print('get_model ok')
PY
```

### 7.5 第五步：再尝试完整 eval

最后再去跑：

```bash
cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05
bash eval.sh <task_name> <task_config> pi05_aloha_full_base model_robotwin <seed> <gpu_id>
```

注意：

- `eval.sh` 最终会调用 `external/RoboTwin/script/eval_policy.py`
- 这个脚本默认 `test_num = 100`
- 所以它不是“快速 smoke test”，而是偏正式评测入口

如果你只是想小试，建议先停在前四步。

---

## 8. 当前建议的使用方式

### 8.1 快速检查模型能不能载入

推荐命令：

```bash
cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/external/RoboTwin:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/src:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/packages/openpi-client/src:$PYTHONPATH
export MPLCONFIGDIR=/tmp

python - <<'PY'
from policy.pi05.deploy_policy import get_model
usr_args = {
    "train_config_name": "pi05_aloha_full_base",
    "model_name": "model_robotwin",
    "checkpoint_id": 30000,
    "pi0_step": 50,
}
model = get_model(usr_args)
print("OK: model loaded", type(model).__name__)
PY
```

### 8.2 正式跑 RoboTwin eval

```bash
cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05

source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final

export PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/external/RoboTwin:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/src:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/packages/openpi-client/src:$PYTHONPATH
export MPLCONFIGDIR=/tmp

bash eval.sh click_bell demo_clean pi05_aloha_full_base model_robotwin 0 0
```

说明：

- `click_bell` 只是一个示例 task 名
- `demo_clean` 是当前可见的 task config
- `0 0` 分别是 `seed` 和 `gpu_id`

但要注意，是否能真正完整跑完，还要看：

- RoboTwin 仿真环境依赖是否齐
- 显卡 / 渲染 / SAPIEN / 资产是否都正常
- 对应 task 是否对当前 embodiment 配置兼容

---

## 9. 常见报错与含义

### 9.1 `No module named 'numpy'`

含义：

- 你没进正确 conda 环境
- 或依赖没装全

优先检查：

```bash
which python
python -c "import numpy; print(numpy.__version__)"
```

### 9.2 `header too large`

含义：

- loader 把 `model.safetensors.index.json` 当权重文件读了

当前代码里已经修过；如果再出现，说明你跑的不是当前修过的代码版本。

### 9.3 `Unexpected keys ... value_head.*`

含义：

- checkpoint 含训练额外头，推理模型没有

当前代码里已经降级为 warning；如果它再次导致 hard fail，说明你跑的还是旧版本。

### 9.4 `FrozenInstanceError: cannot assign to field 'asset_id'`

含义：

- 你跑到旧版 `policy_config.py`

当前代码里已改成 `dataclasses.replace(...)`。

### 9.5 tokenizer 路径不存在

如果报 tokenizer 相关文件找不到，优先检查：

- 本地 `paligemma_tokenizer.model` 是否存在
- OpenPI 的 tokenizer 资源是否下载齐
- 当前机器路径是否仍沿用旧路径

---

## 10. 当前改动文件

本次为了打通 RoboTwin 下的 PI05，改过的核心文件有：

- `external/RoboTwin/policy/pi05/src/openpi/models/model.py`
- `external/RoboTwin/policy/pi05/src/openpi/policies/policy_config.py`

如果后续你要迁移到别的分支、别的机器，最容易丢失的就是这两处兼容修改。

---

## 11. 下一步建议

建议按下面顺序继续：

1. 先用本文档第 8.1 节命令确认模型加载稳定
2. 再跑一次最轻量的 `eval.sh`
3. 如果完整 eval 太重，先把 `eval_policy.py` 改成支持 `--test_num 1`
4. 再做真正的一回合 smoke test
5. 最后再跑正式评测

如果你希望后续维护更轻松，建议再做两个小改动：

- 给 `external/RoboTwin/script/eval_policy.py` 增加 `--test_num` CLI 参数
- 给 `policy/pi05` 增加一个单独的 `smoke_test.sh`

这样以后就不用每次都跑默认 100 次正式评测。

---

## 12. 单回合录视频（smoke test）

为了避免 `eval_policy.py` 默认一次跑 100 回合，现在仓库已经加了一个单回合入口：

- `scripts/robotwin_pi05_smoke_eval.sh`

默认行为：

- 使用 `demo_clean`
- 使用 `pi05_aloha_full_base`
- 使用 `model_robotwin`
- 使用 `seed=0`
- 只跑 `test_num=1`

### 12.1 录一个任务的单回合 demo

示例：

```bash
bash scripts/robotwin_pi05_smoke_eval.sh pick_dual_bottles 0
```

完整参数形式：

```bash
bash scripts/robotwin_pi05_smoke_eval.sh \
  <task_name> \
  <gpu_id> \
  [task_config] \
  [train_config_name] \
  [model_name] \
  [seed] \
  [test_num]
```

例如：

```bash
bash scripts/robotwin_pi05_smoke_eval.sh \
  place_can_basket \
  0 \
  demo_clean \
  pi05_aloha_full_base \
  model_robotwin \
  0 \
  1
```

### 12.2 一次性录这四个任务

仓库还新增了批量脚本：

- `scripts/robotwin_pi05_record_four_demos.sh`

它会顺序跑：

- `pick_dual_bottles`
- `pick_diverse_bottles`
- `place_object_basket`
- `place_can_basket`

直接使用：

```bash
bash scripts/robotwin_pi05_record_four_demos.sh 0
```

完整参数形式：

```bash
bash scripts/robotwin_pi05_record_four_demos.sh \
  [gpu_id] \
  [task_config] \
  [train_config] \
  [model_name] \
  [seed] \
  [test_num]
```

例如：

```bash
bash scripts/robotwin_pi05_record_four_demos.sh \
  0 \
  demo_clean \
  pi05_aloha_full_base \
  model_robotwin \
  0 \
  1
```

### 12.3 视频输出路径

视频会被 RoboTwin 自动写到：

```bash
external/RoboTwin/eval_result/<task_name>/pi05/<task_config>/<ckpt_setting>/<timestamp>/episode0.mp4
```

例如：

```bash
external/RoboTwin/eval_result/pick_dual_bottles/pi05/demo_clean/model_robotwin/<timestamp>/episode0.mp4
```

---

## 13. 当前已知限制：SAPIEN 渲染设备

当前这次调试里，模型链路已经跑通，但本机会话抓到的渲染错误是：

```text
RuntimeError: failed to find a rendering device
```

这代表：

- 模型本身不是当前阻塞
- 当前阻塞在 `SAPIEN renderer` 初始化
- 没有可用图形/渲染设备时，无法录制可视化视频

### 13.1 先验证 renderer 能否初始化

在目标会话里先跑：

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
cd /nfs_global/S/yangrongzheng/pi05/external/RoboTwin

PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/external/RoboTwin:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/src:/nfs_global/S/yangrongzheng/pi05/external/RoboTwin/policy/pi05/packages/openpi-client/src:$PYTHONPATH \
python - <<'PY'
import sapien.core as sapien
from sapien.render import set_global_config
engine = sapien.Engine()
set_global_config(max_num_materials=50000, max_num_textures=50000)
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene(sapien.SceneConfig())
print('render ok')
PY
```

只有这一步通过后，再去跑录视频脚本才有意义。

---

## 6. 这次额外修掉的控制对齐问题

在把 `pi05` 权重正式接进 RoboTwin 时，发现还有一层更隐蔽的问题：

- 模型能加载
- checkpoint 也能读
- 但 rollout 时控制表现不对，像是“动作方向 / gripper 空间不匹配”

最终定位到根因是：**RoboTwin 这条 Aloha 输入输出链路需要启用 `adapt_to_pi=True`，否则动作会被按错误空间解释。**

### 6.1 根因说明

`openpi` 的 `AlohaInputs / AlohaOutputs` 会在 `adapt_to_pi=True` 时自动做两类关键变换：

- 对若干关节做符号翻转
- 对 gripper 做 Aloha ↔ PI 的编码/反编码变换

如果这里是 `False`，会出现：

- 部分关节方向反了
- gripper 开合范围映射不对
- 最终表现为 RoboTwin 中 policy “能出动作，但控制不对齐”

### 6.2 已修改文件

这次已经把 RoboTwin 相关 config 全部统一改成 `adapt_to_pi=True`：

- `external/RoboTwin/policy/pi05/src/openpi/training/config.py`
- `external/RoboTwin/policy/pi05/pi_model.py`

具体包括：

- `pi05_aloha_full_base`
- `pi0_base_aloha_robotwin_lora`
- `pi0_fast_aloha_robotwin_lora`
- `pi0_base_aloha_robotwin_full`
- `pi0_fast_aloha_robotwin_full`

这样之后，无论是默认 `pi05_aloha_full_base`，还是 fallback 到 `pi0_*_aloha_robotwin_*`，都会走正确动作空间。

### 6.3 当前建议用法

保持当前命令即可：

```bash
bash scripts/robotwin_pi05_smoke_eval.sh adjust_bottle 0 demo_clean pi05_aloha_full_base model_robotwin 0 1
```

对应 checkpoint 路径：

```bash
external/RoboTwin/policy/pi05/checkpoints/pi05_aloha_full_base/model_robotwin/30000
```

如果默认 config 不存在，`pi_model.py` 里已经加入 fallback 逻辑，会尽量落到 RoboTwin 专用 config，而不是继续沿用错误动作定义。

---

## 7. 为什么当前机器上还没拿到真实成功 rollout

这次我已经确认：

- `pi05` 权重加载链路是通的
- 动作空间错位问题已经修掉

但当前这台机器上，**完整 RoboTwin rollout 仍被宿主图形/渲染能力卡住**。

### 7.1 现象

运行：

```bash
cd external/RoboTwin
python script/test_render.py
```

会直接打印：

```text
Render Error
```

继续抓异常后，真实错误是：

```text
RuntimeError: failed to find a rendering device
```

### 7.2 这意味着什么

这说明当前问题已经不是 `pi05` policy 对齐问题，而是：

- 当前运行环境里没有可用的 SAPIEN/Vulkan 渲染设备
- 或者 Vulkan ICD 没正确暴露给这个会话
- 导致 RoboTwin 环境本身无法正常建 scene / renderer

在这种情况下：

- 模型可以 load
- config 可以对齐
- 但真实任务 rollout 无法作为最终成功证据跑出来

### 7.3 需要什么宿主条件

要真正跑出 `adjust_bottle` / `pick_dual_bottles` 这类 RoboTwin 任务，宿主至少需要：

- 可用 GPU 图形驱动
- SAPIEN 可见的 Vulkan 设备
- 正确的 Vulkan ICD（例如 `VK_ICD_FILENAMES` 指向有效 json）

如果是在服务器/容器里跑，常见需要检查：

```bash
/etc/vulkan/icd.d/
/usr/share/vulkan/icd.d/
VK_DRIVER_FILES
VK_ICD_FILENAMES
```

### 7.4 结论

所以当前仓库状态可以分成两部分看：

1. **代码接入层面：已经修好**
   - 权重可识别
   - 模型可加载
   - 动作空间已对齐

2. **宿主执行层面：当前机器仍缺渲染设备**
   - 需要换到有图形/Vulkan 能力的宿主，或补齐 Vulkan/驱动配置
   - 然后再跑同样的 smoke/eval 命令拿真实成功率

---

## 8. value model 训练推进记录（2026-06-05）

本节记录 `recap_workspace` 下 value model 训练链路的**最新真实进度**，避免和前面的 RoboTwin rollout 结果混淆。

### 8.1 本次确认的脚本入口

本次 value model 训练入口是：

```bash
recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm
```

它最终调用：

```bash
recap_workspace/run_value_local.py
```

再进入：

```bash
recap_workspace/vendor/rlinf-recap/examples/recap/value/train_value.py
```

这条链路训练的是 **value model**，不是 RoboTwin policy rollout。

### 8.2 这次新推进到哪一步

相比此前 `recap-adj100-763373.err` / `recap-adj100-763379.err` 里卡住的：

- `TypeError: expected bytes, PosixPath found`

这次通过本地最小复现，已经确认训练入口**明显推进到了更后面的阶段**：

1. `run_value_local.py` 能执行到：
   - `patch_sys_path_for_vendor_rlinf`
   - `patch_ray_init_pathlike`
   - `patch_ray_coreworker_pathlike`
   - `patch_worker_init`
2. 能真正进入：

```bash
recap_workspace/vendor/rlinf-recap/examples/recap/value/train_value.py
```

3. 旧的 `PosixPath found` 不再是当前最先撞到的报错。

这说明此前针对：

- vendored `rlinf/openpi/libero` 优先级
- `ray.init(...)` path-like 参数
- `ray._raylet.CoreWorker(...)` path-like 参数

的 patch，已经让训练链路继续向前推进。

### 8.3 本次遇到的新卡点

本次推进过程中，按顺序又暴露出两个新的真实问题：

#### 问题 A：解释器没有拿到 `ray`

早期一次本地最小复现中，先撞到：

- `ModuleNotFoundError: No module named 'ray'`

根因是：

- `run_value_local.py` 执行时，`sys.path` 虽然清理了外部路径，但没有保证 `RLINF_VENV_SITE_PACKAGES` 永远在最前面

已做处理：

- `recap_workspace/run_value_local.py` 现在会显式把 `RLINF_VENV_SITE_PACKAGES` 放进 `preferred_paths`

处理后，训练已能继续前进，不再先死在 `No module named 'ray'`。

#### 问题 B：Hydra 缺 `REPO_PATH`

之后新的本地最小复现里，又撞到：

- `KeyError: Environment variable 'REPO_PATH' not found`
- 最终表现为 `hydra.errors.ConfigCompositionException`

这说明：

- 训练配置里依赖 `REPO_PATH`
- 只要把 slurm 中已有的 `REPO_PATH` 带上，链路就还能继续前进

在补上：

```bash
export REPO_PATH=/nfs_global/S/yangrongzheng/pi05/recap_workspace/vendor/rlinf-recap
```

之后，训练再次继续向前推进。

#### 问题 C：当前交互环境里 Ray 建 socket 被拒绝

在再往后一轮本地最小复现里，当前新的最先卡点变成：

- `PermissionError: [Errno 1] Operation not permitted`

触发位置在：

- `ray._private.services.get_node_ip_address()`
- `socket.socket(AF_INET, SOCK_DGRAM)`

也就是说，现在 value model 训练链路的最新阻塞点已经变成：

- **当前交互环境不允许 Ray 在这里按默认方式探测 node IP / 建 socket**

这不是前面 RoboTwin policy 本身的兼容问题，而是：

- 当前环境下 Ray 初始化的系统权限 / 运行环境约束问题

### 8.4 截至当前的真实状态

截至 2026-06-05 本次更新，可以诚实确认的结论是：

- **RoboTwin pi05 rollout：已跑通**
- **value model 训练：未最终跑通，但已经从旧的 `PosixPath found` 推进到了新的 Ray socket 权限问题**

也就是：

- 旧问题：`TypeError: expected bytes, PosixPath found`
- 新问题：`PermissionError: [Errno 1] Operation not permitted`

这说明不是原地踏步，而是确实已经推进到新的阶段。

### 8.5 还差什么才算“value model 训练跑通”

要把这条链路真正标记为“跑通”，至少还需要一份新的真实日志，证明以下任一项：

- 已成功完成 `Cluster(...)` / `ray.init(...)` 初始化
- 已成功创建 value training workers
- 已开始真实 training step（例如出现 step / loss / checkpoint 保存）

在拿到这些真实证据之前，不能把 value model 训练写成“已跑通”。

### 8.6 当前 slurm 入口本身已经补齐的关键环境项

需要特别说明的是，当前正式 value 训练脚本：

```bash
recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm
```

本身已经具备这些关键环境设置：

- `ulimit -n 8192`
- `REPO_PATH=${ROOT}/vendor/rlinf-recap`
- `OPENPI_LOCAL_PYTHON` 由 `scripts/use_local_openpi_env.sh` 导出
- 独立的 `RAY_TMPDIR`
- 显式 `RAY_ADDRESS=127.0.0.1:<port>`
- 显式 `ray start --head --node-ip-address=127.0.0.1`

所以截至当前，更合理的判断是：

- **slurm 脚本入口本身并不明显缺关键环境变量**
- 当前真实阻塞更像是：
  - 交互执行环境下对 Ray 本地 socket / 端口分配的权限限制
  - 而不是 slurm 文件里少写了某个显而易见的变量

换言之，如果后续要继续验证“是否真正能跑通 value model 训练”，
最有价值的新证据仍然是：

- 在真正的集群 batch 环境里重新提交一次
- 查看新的 `recap-adj100-<jobid>.out/.err`
- 确认是否已经进入 training step / loss / checkpoint 保存

### 8.7 对“是否已偷偷跑通”的补充审计

本次还额外检查了：

- `recap_workspace/logs/` 下最近生成的目录
- 本地最小复现生成的 `recap_adjust_bottle_manual_verify*` 目录

审计结果：

- 目前只看到新的目录创建
- **没有发现新的训练 step / loss / checkpoint / worker log 成功产物**
- 也没有发现一份新的 `recap-adj100-<jobid>.out/.err` 能证明 value model 训练已经真正启动完成

因此，可以排除一种误判：

- 不是“其实已经成功了，只是前面没注意到产物”

截至当前，最保守且真实的状态仍然是：

- value model 训练**尚未最终跑通**
- 但已经被推进到当前环境的 Ray socket / 端口权限限制这一步

### 8.8 新的真实 batch 证据：job 763625

在真正的集群 batch 环境里，已经重新提交：

```bash
sbatch recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm
```

获得 job：

- `763625`

这次新的真实日志说明两件事：

#### 1）之前在交互环境里卡住的 Ray socket / 端口权限问题，不是 batch 环境里的主问题

`recap_workspace/logs/recap-adj100-763625.out` 里已经出现：

- `Local node IP: 10.200.198.55`
- `Ray runtime started.`

这说明：

- 在真正的 slurm/batch 环境中，Ray head 可以正常启动
- 先前在交互环境里反复遇到的 `PermissionError: [Errno 1] Operation not permitted`，并不是这条 value 训练链在 batch 环境里的最终主阻塞

#### 2）当前新的 batch 主阻塞点已经进一步收敛为 Python 环境混装问题

这次 `763625` 的 `.err` 里出现：

- `ModuleNotFoundError: No module named 'huggingface_hub'`

触发链路大意是：

- `run_value_local.py` 顶层 import `lerobot_dataset`
- `lerobot_dataset` 从 `RLinf-main/.venv` 里的 `datasets` 包导入
- 该 `datasets` 再去 import `huggingface_hub`
- 但当前 worker 看到的 Python 包路径组合中，没有把 `huggingface_hub` 解析成功

需要特别注意的是：

- 当前 `.conda-pi05-openpi-final` 里其实安装了 `huggingface_hub`
- 当前 `.conda-pi05-openpi-final` 里也能 `import datasets`
- 但 `763625` 的报错路径显示，它真正 import 到的 `datasets` 来自：

```bash
/nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages/datasets
```

这说明当前新问题更像是：

- **conda 环境与 RLinf venv 的 site-packages 混用后，依赖解析不一致**
- 而不是 “机器上根本没有安装 `huggingface_hub`”

### 8.9 截至当前的最新真实状态

最新真实链路已经推进到：

- slurm 作业能提交
- Ray head 能正常启动
- value 训练入口已经真正进入 batch 环境执行
- 当前新的第一阻塞点变成：
  - `ModuleNotFoundError: No module named 'huggingface_hub'`

因此截至本次更新，最准确的结论是：

- **value model 训练仍未最终跑通**
- 但主阻塞点已从：
  - `PosixPath found`
  - `ray` 缺失
  - `REPO_PATH` 缺失
  - 交互环境 socket 权限
- 进一步收敛成：
  - **batch 环境里的 Python 依赖/路径混装问题（当前体现为 `huggingface_hub` 缺失）**

### 8.10 job 763627：修复顶层 lerobot import 后的新结果

在把 `run_value_local.py` 里顶层 `lerobot_dataset` import 延后之后，又重新提交了：

- job `763627`

新日志 `recap_workspace/logs/recap-adj100-763627.out` 表明：

- Ray head 仍可正常启动
- `run_value_local.py` 已执行到：
  - `patch_sys_path_for_vendor_rlinf`
  - `patch_pathlib_for_ray`
  - `patch_ray_init_pathlike`
  - `patch_ray_coreworker_pathlike`
  - `patch_ray_node_ip`
  - `patch_hf_datasets_cache`

新的 `.err` 表明：

- 当前最早报错已进一步精确收敛到：
  - `_patch_hf_datasets_cache()` 中 `import datasets.config as ds_config`
- 而这个 `datasets` 实际来自：
  - `RLinf-main/.venv/lib/python3.11/site-packages/datasets`
- 但当它去依赖：
  - `huggingface_hub`
- 当前解释器看到的 `sys.path` 组合里，仍未解析成功

所以目前更准确的根因描述是：

- 不是单纯“lerobot 顶层 import 太早”
- 而是 **`run_value_local.py` 的 `sys.path` 优先级仍会把 `datasets` 指到 RLinf venv，但没有同时保证它的依赖链（如 `huggingface_hub`）一致可见**

为进一步修正这个问题，本次又把：

- 当前解释器自己的 conda `site-packages`

显式加入 `preferred_paths` 的最前面，尝试确保：

- `huggingface_hub` 这类只装在 conda 环境里的包
- 与 `RLinf_VENV_SITE_PACKAGES` 中的依赖一起可见

截至当前，value model 训练仍未最终跑通，但主阻塞点已进一步收敛到：

- **`datasets` / `huggingface_hub` 的路径优先级与依赖可见性一致性问题**

### 8.11 最新修正：不再用 `OPENPI_LOCAL_PYTHON` 跑 value 训练入口

经过 `763625` / `763627` / `763628` 这三轮真实 batch 证据，可以更明确判断：

- 当前 `OPENPI_LOCAL_PYTHON` 指向的是：

```bash
/nfs_global/S/yangrongzheng/RLinf-main/.venv/bin/python
```

- 但这套解释器与当前实际可见的依赖链并不稳定一致
- 尤其在导入：
  - `datasets`
  - `huggingface_hub`
  - `lerobot`
  - `ray`

时，会出现 **conda 环境与 RLinf venv 的 site-packages 混装不一致** 问题

因此本次进一步把：

```bash
recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm
```

中的入口解释器，从：

```bash
"${OPENPI_LOCAL_PYTHON:-python}"
```

改成了强制使用：

```bash
/nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final/bin/python
```

核心思路是：

- 让运行时解释器、已安装依赖、`huggingface_hub` 可见性先稳定下来
- 再通过 `PYTHONPATH` 接入 vendored `rlinf/openpi/libero`
- 尽量避免“解释器是 RLinf venv，依赖却又部分来自 conda”的混装问题

截至本文档这次更新，这个修正**已落到脚本中**，但还需要新的 batch 作业日志来证明：

- 是否已经越过 `ModuleNotFoundError: huggingface_hub`
- 是否进一步进入真实训练 step / loss / checkpoint

### 8.12 job 763630：已越过 `huggingface_hub`，新卡点变成 Ray actor 内的 `rlinf` 包解析

在把 value 训练入口强制改成 conda Python 之后，再次提交了：

- job `763630`

这次新的真实日志说明：

#### 1）`huggingface_hub` 问题已经被越过

`763630.out/.err` 显示：

- 已成功执行 `_patch_hf_datasets_cache()`
- 已成功继续到：
  - `patch_lerobot_timestamp_sync`
  - `patch_lerobot_version_compatibility`
  - `patch_lerobot_query_hf_dataset`
  - `patch_lerobot_metadata_version`
  - `patch_worker_init`
- 并真正进入：
  - `train_value.py`
  - `validate_cfg(cfg)`
  - `Cluster(...)`
  - `NodeProbe(...)`

这说明：

- 之前反复阻塞的 `ModuleNotFoundError: No module named 'huggingface_hub'`
- 已经不是当前最先出现的问题

#### 2）当前新的主阻塞点

现在新的最早报错变成：

- `ray.exceptions.ActorDiedError`

根因更底层是：

- `ModuleNotFoundError: No module named 'rlinf.scheduler.cluster.config'; 'rlinf.scheduler.cluster' is not a package`

触发位置说明：

- 主进程已能 import `rlinf.scheduler.cluster`
- 但 Ray actor 子进程在创建 `NodeProbe` 相关 actor 时，
  对 `rlinf.scheduler.cluster` 的解析结果不一致
- 于是把它当成了一个普通模块，而不是 package
- 进一步导致：

```bash
from .config import ClusterConfig, NodeGroupEnvConfig
```

这类包内相对导入在 actor 进程中失败

#### 3）当前最准确的状态

截至 job `763630` 的真实证据：

- value model 训练**仍未最终跑通**
- 但链路已经推进到：
  - Ray 正常启动
  - Hydra 配置解析通过
  - value training 主入口启动
  - Cluster 初始化开始
  - actor 创建阶段
- 当前最核心的新根因，已经收敛到：
  - **Ray actor 子进程中的 `rlinf` 包解析 / import 语义冲突**

这比此前的环境缺包问题更接近“训练真正开始”。

### 8.13 针对 `763630` 新根因的修正：避免 Ray actor 内相对导入把 `cluster` 解析成普通模块

根据 `763630.err` 的最新报错：

- `ModuleNotFoundError: No module named 'rlinf.scheduler.cluster.config'; 'rlinf.scheduler.cluster' is not a package`

本次进一步判断为：

- 在主进程里，`rlinf.scheduler.cluster` 作为 package 是正常的
- 但在 Ray actor 子进程里，包内相对导入：

```python
from .cluster import Cluster
from .cluster import Cluster, ClusterEnvVar
```

有机会把 `cluster` 解析成一个普通模块对象，而不是 `rlinf.scheduler.cluster` 这个 package 上下文

因此本次在：

```bash
recap_workspace/vendor/rlinf-recap/rlinf/scheduler/cluster/node.py
```

里把相关导入改成了**绝对导入**：

```python
from rlinf.scheduler.cluster.cluster import Cluster
from rlinf.scheduler.cluster.cluster import Cluster, ClusterEnvVar
```

目的是降低：

- Ray actor 反序列化 / 远端 import 时
- `cluster` 名称被误解析成模块而非 package 的风险

截至当前，这个修正已经落到代码里，但还需要新的 batch 日志来验证：

- 是否已经越过 `rlinf.scheduler.cluster.config` 这个包解析错误
- 是否进一步进入 worker 创建 / training step / loss / checkpoint

### 8.14 最新修复：启动时显式清理 `sys.modules` 中的 `rlinf.scheduler.cluster*`

由于 `763630` / `763635` 都稳定失败在：

- `ModuleNotFoundError: No module named 'rlinf.scheduler.cluster.config'; 'rlinf.scheduler.cluster' is not a package`

当前进一步怀疑：

- 在主进程或 Ray 远端 worker 反序列化过程中
- `sys.modules['rlinf.scheduler.cluster']` 很早就被注册成了错误的普通模块对象
- 之后即使文件系统上 `rlinf/scheduler/cluster/` 目录结构是完整 package，
  Python 也会优先复用这个错误缓存

因此本次在：

```bash
recap_workspace/run_value_local.py
```

里进一步做了显式清理：

- `rlinf.scheduler.cluster`
- `rlinf.scheduler.cluster.*`

这样做的目的，是尽量避免：

- 旧的错误模块对象残留在 `sys.modules`
- 导致后续 Ray worker / actor 仍然把 `cluster` 当成普通模块而非 package

截至当前，这个修复已落到代码里，但还需要下一轮真实 batch 日志验证是否生效。

### 8.15 新尝试：给 Ray worker 增加 preload 模块，预热 `rlinf.scheduler.cluster` package

由于 `763630` / `763635` / `763636` 都稳定表明：

- 主进程内 `rlinf.scheduler.cluster` 可以正常工作到很后面
- 但 Ray actor 远端进程里仍会把它解析错成“不是 package”

因此本次新增了一个本地模块：

```bash
recap_workspace/ray_worker_preload.py
```

它会在模块导入时：

- 重新整理 `sys.path`
- 显式把 conda `site-packages`、RLinf venv `site-packages`、vendored `rlinf/openpi/libero` 放到前面
- 清理 `sys.modules` 中的：
  - `rlinf.scheduler.cluster`
  - `rlinf.scheduler.cluster.*`
- 然后主动预热导入：
  - `rlinf.scheduler.cluster`
  - `rlinf.scheduler.cluster.config`
  - `rlinf.scheduler.cluster.node`

同时在：

```bash
recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm
```

里新增：

```bash
export RAY_WORKER_PRELOAD_MODULES="recap_workspace.ray_worker_preload"
```

设计目标是：

- 让 Ray worker / actor 远端进程在真正执行任务前，
  先以一致的 package 语义把 `rlinf.scheduler.cluster` 预热好
- 尝试解决当前最核心的：
  - `rlinf.scheduler.cluster is not a package`

截至当前，这个修正刚落地，还需要下一轮真实 batch 日志来验证是否真正生效。

### 8.16 截至当前的阶段性结论（必须诚实保留）

截至本轮所有真实 batch 验证完成后，必须保留以下结论：

- **RoboTwin rollout 已跑通**
- **value model 训练仍未跑通**

最新一轮真实 batch 验证已经覆盖到：

- `763625`
- `763627`
- `763628`
- `763630`
- `763635`
- `763636`
- `763639`

这些日志已经足以说明：

1. `sbatch` 提交本身正常
2. Ray head 启动本身正常
3. `train_value.py` 主入口已经被真实执行
4. 当前链路并不是卡在最初级的环境问题
5. 当前稳定主阻塞已经收敛到：

```bash
ModuleNotFoundError: No module named 'rlinf.scheduler.cluster.config'; 'rlinf.scheduler.cluster' is not a package
```

因此，在拿到以下任一真实证据之前，都**不能**把 value model 训练写成“已跑通”：

- actor 创建成功并稳定存活
- training step / loss 输出
- checkpoint 保存产物

换句话说，截至当前最准确、最诚实的状态只有一句：

- **value model 训练尚未跑通，当前主根因是 Ray actor 侧 `rlinf.scheduler.cluster` 的 package/module 解析冲突**

### 8.17 最新环境收敛方向：优先当前仓库自己的 `.venv_recap311`

结合 `../RLinf-main` 与 `../RLinf-recap` 的结构对比，可以确认：

- `../RLinf-main` 仍保留旧式单文件：

```bash
rlinf/scheduler/cluster.py
```

- 当前 vendored / `../RLinf-recap` 使用的是新式 package：

```bash
rlinf/scheduler/cluster/
```

而当前最新主错误：

- `rlinf.scheduler.cluster ... is not a package`

与两套结构语义混用高度一致。

此外，当前仓库自己的：

```bash
recap_workspace/.venv_recap311
```

已经具备 value 训练所需的关键依赖：

- `ray`
- `hydra`
- `omegaconf`
- `datasets`
- `huggingface_hub`

因此本次进一步把环境收敛方向改成：

- **优先使用当前仓库自己的 `.venv_recap311`**
- 尽量避免再借任何外部 `RLinf-main` 的环境语义

具体改动：

- `scripts/use_local_openpi_env.sh` 中把 `PATH` 的优先级改成：
  1. `recap_workspace/.venv_recap311/bin`
  2. `.conda-pi05-openpi-final/bin`

目的是：

- 让默认命令解析更倾向当前仓库本地的 recap venv
- 进一步隔离旧版 `RLinf-main` 可能残留的单模块 `cluster.py` 结构语义

截至当前，这个修正已落到当前仓库，但仍需要新的真实 batch 日志来验证：

- 是否能真正打掉 `rlinf.scheduler.cluster ... is not a package`

### 8.18 最新主根因收敛：Ray 集群进程与训练进程 Python 版本不一致

在 job `763646` 中，新的真实 batch 证据把主问题进一步收敛成了：

- `RuntimeError: Version mismatch`

具体表现为：

- Ray cluster 启动时使用：
  - Python `3.11.10`
- 训练进程连接时使用：
  - Python `3.11.14`

这说明当前问题已经不是最早的：

- `PosixPath found`
- `huggingface_hub` 缺失
- `rlinf.scheduler.cluster` 包冲突

而是更底层也更直接的：

- **Ray head / worker / driver 没有使用同一套 Python 解释器**

为避免这种版本不一致，本次进一步把：

```bash
recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm
```

中的以下动作全部统一到：

```bash
recap_workspace/.venv_recap311/bin/python
```

包括：

- `ray stop`
- `ray start --head`
- `run_value_local.py`

并显式设置：

```bash
export OPENPI_LOCAL_PYTHON="${ROOT}/.venv_recap311/bin/python"
```

这一步的目的很明确：

- 保证 Ray cluster 进程和 value training 进程看到的是同一套 Python 版本与依赖
- 避免再次出现：
  - cluster 用 `3.11.10`
  - driver/worker 用 `3.11.14`

截至当前，这个修正已经落到脚本，但还需要新的真实 batch 日志来验证：

- 是否已经越过 Python version mismatch
- 是否进一步进入 actor 创建成功 / training step / loss / checkpoint


### 8.19 最新真实进展：已打掉 `RLinf-main` 环境污染与 `cluster is not a package`

在继续对比以下几套代码结构后：

- `../RLinf-main`
- `../RLinf-recap`
- `../RLinf-Pi05-LIBERO-SFT`
- `recap_workspace/vendor/rlinf-recap`

确认了一个关键事实：

- `../RLinf-main` 仍是旧结构：
  - `rlinf/scheduler/cluster.py`
- 当前 vendored 的 `rlinf-recap` 是新结构：
  - `rlinf/scheduler/cluster/`

这与之前长期稳定出现的错误完全吻合：

- `ModuleNotFoundError: No module named 'rlinf.scheduler.cluster.config'; 'rlinf.scheduler.cluster' is not a package`

进一步检查 `763647` 的 Ray 启动命令后，发现虽然 worker 解释器已经切到：

- `recap_workspace/.venv_recap311/bin/python`

但 Ray 仍在从旧环境路径启动内部脚本：

- `/nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages/ray/...`

这说明此前真正的污染源不是单纯 `PYTHONPATH`，而是：

- **Ray 自身脚本和 site-packages 仍来自外部 `RLinf-main` 环境**

因此本轮新增了两类硬隔离修复：

1. `scripts/use_local_openpi_env.sh`
   - 显式删除：
     - `/nfs_global/S/yangrongzheng/RLinf-main`
     - `/nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages`
     - `/nfs_global/S/yangrongzheng/RLinf-main/RLinf_deps/libero`
     - `/nfs_global/S/yangrongzheng/RLinf-main/.venv/bin`
   - 并把 `PATH` 强制收敛到：
     - `recap_workspace/.venv_recap311/bin`
     - `.conda-pi05-openpi-final/bin`

2. `recap_workspace/scripts/recap_value_adjust_bottle_100_1gpu.slurm`
   - 不再 `conda activate`
   - 改为直接 `source scripts/use_local_openpi_env.sh`
   - 再显式设置：
     - `OPENPI_LOCAL_PYTHON=${ROOT}/.venv_recap311/bin/python`
     - 极简 `PATH`
     - `unset PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE`
   - 并在 job 启动时打印：
     - `PATH`
     - `PYTHONPATH`
     - `sys.executable`
     - `ray.__file__`
     - `sys.path[:8]`

另外在 `recap_workspace/run_value_local.py` 中，也进一步收紧了：

- 不再把 `conda` 的 `site-packages` 提到最前面
- `preferred_paths` 只保留当前仓库自己的：
  - `recap_workspace/.venv_recap311/lib/python3.11/site-packages`
  - `openpi_official/src`
  - `recap_workspace/vendor/rlinf-recap`
  - `recap_workspace/vendor/libero`
  - 当前仓库根目录
- 额外排除：
  - `../RLinf-main`
  - `../RLinf-recap`
  - `../RLinf-Pi05-LIBERO-SFT`

### 8.20 新 batch 证据：job `763652` 已越过旧 `cluster` 冲突，进入 worker init

提交新 job：

- `763652`

从 `recap_workspace/logs/recap-adj100-763652.out` 可确认：

- `OPENPI_LOCAL_PYTHON=/nfs_global/S/yangrongzheng/pi05/recap_workspace/.venv_recap311/bin/python`
- `ray.__file__=/nfs_global/S/yangrongzheng/pi05/recap_workspace/.venv_recap311/lib/python3.11/site-packages/ray/__init__.py`
- `PYTHONPATH` 中已不再出现 `RLinf-main`
- `Ray runtime started.` 正常
- `train_value.py` 正常执行
- `Cluster(...)` / `NodeProbe(...)` / `WorkerGroup` 初始化都已成功越过
- `ActorGroup` 已真正创建成功
- `FSDPValueSftWorker.init_worker` 已开始执行
- 已进入：
  - `build_dataloader`
  - tokenizer 加载
  - train dataset entries 构建

这说明此前的核心阻塞：

- `rlinf.scheduler.cluster ... is not a package`

已经被这轮环境隔离真实打掉。

### 8.21 当前新的主阻塞：LeRobot 数据集版本兼容异常在 worker 侧未被正确补丁

`763652` 的最新失败点已经变成：

- `lerobot.common.datasets.backward_compatibility.BackwardCompatibilityError`

关键日志显示在 worker 里执行：

- `LeRobotDatasetMetadata(local_path.name, root=local_path)`
- `check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)`
- 然后抛出：
  - `BackwardCompatibilityError(repo_id, v_check)`

而 driver 侧最终看到的是：

- `TypeError: BackwardCompatibilityError.__init__() missing 1 required positional argument: 'version'`
- `RuntimeError: Failed to unpickle serialized exception`

这说明：

- 不是 Ray actor 没创建成功
- 不是旧的 `cluster package/module` 冲突
- 而是 **worker 里真实触发了 LeRobot 数据集版本兼容报错，Ray 在反序列化该异常时又出现了兼容性问题**

结合当前 `run_value_local.py` 中已有 patch：

- `_patch_lerobot_version_compatibility()`
- `_patch_lerobot_metadata_version()`

可以推断出新的剩余问题是：

- **这些 LeRobot 兼容 patch 目前只在 driver 进程生效，尚未在远端 Ray worker 里完整生效**

因此截至当前：

- `RoboTwin rollout`：已跑通
- `value model training`：仍未完全跑通
- 但当前状态已经比之前前进很多，真实地进入了：
  - worker actor 创建成功
  - value worker 初始化
  - dataloader / dataset metadata 路径

还缺的最终成功证据仍然是：

- 正式进入 training step / loss
- checkpoint 产物


### 8.22 新验证：worker 侧 LeRobot patch 已命中，主错误继续后移

在 `763653` 中，我尝试把 LeRobot 的兼容 patch 注入到：

- `recap_workspace/ray_worker_preload.py`

但真实 batch 日志表明，这还不足以完全覆盖实际抛异常的 worker 执行路径；worker 里仍然继续触发：

- `BackwardCompatibilityError`

因此进一步把兼容 patch 直接下沉到真正构造数据集的路径：

- `recap_workspace/vendor/rlinf-recap/rlinf/data/datasets/recap/value_model.py`

即在 `ValueDataset.__init__` 中、调用：

- `LeRobotDatasetMetadata(local_path.name, root=local_path)`

之前，先显式执行本地兼容 patch。

这轮新的真实 batch 作业：

- `763654`

给出了比之前更进一步的证据：

- worker actor 仍然成功创建
- 已进入 `ValueDataset.__init__`
- 且这次新的报错不再是原来的 `BackwardCompatibilityError` 本体
- 而是 patch 继续向前推进后暴露出的更具体错误：
  - `AttributeError: property '_version' of 'LeRobotDatasetMetadata' object has no setter`

这说明：

- 当前补丁**已经命中了真正的 worker 数据集代码路径**
- 老的 `cluster package/module` 冲突仍然保持已解决状态
- 但 LeRobot metadata 兼容 patch 里对：
  - `self._version = ...`
  的赋值方式不适用于当前 `LeRobotDatasetMetadata` 实现，因为 `_version` 是只读 property

因此主阻塞再次收敛为一个更小、更直接的兼容点：

- **需要避免给 `LeRobotDatasetMetadata._version` 直接赋值，只保留对 `info['codebase_version']` 的补齐**

截至当前：

- `RoboTwin rollout`：已跑通
- `value model training`：仍未跑通
- 但真实链路已经进一步推进到了：
  - Ray worker 创建成功
  - worker 进入 `ValueDataset` 构造
  - worker 命中本地 LeRobot 数据兼容补丁

仍缺的最终完成证据仍是：

- training step / loss
- checkpoint 产物


### 8.23 新验证：已越过 metadata 兼容，当前主阻塞后移到时间戳同步检查

在修正 `LeRobotDatasetMetadata._version` 只读属性问题后，继续提交新 job：

- `763655`

这次真实日志继续证明链路向前推进了：

- 不再出现：
  - `cluster ... is not a package`
  - `BackwardCompatibilityError`
  - `_version has no setter`
- worker 已进入更深的 `LeRobotDataset(...)` 初始化路径
- 已构造：
  - `LeRobotDatasetMetadata(...)`
  - `ValueDataset._base = LeRobotDataset(...)`

当前新的真实失败点变成：

- `ValueError: One or several timestamps unexpectedly violate the tolerance inside episode range.`

并且这个错误发生在：

- `lerobot.common.datasets.utils.check_timestamps_sync`

对应真实堆栈表明：

- 不是 patch 没命中 `ValueDataset`
- 而是 **`LeRobotDataset` 初始化内部再次触发了时间戳同步校验，而此前 driver 侧的 timestamp patch 还没有覆盖到这一层实际调用路径**

这进一步说明：

- 当前 value training 链路已经明显更深入
- 主阻塞不再是环境或 Ray actor 结构问题
- 而是更具体的数据兼容问题：
  - **LeRobot 数据集时间戳同步检查需要在实际 dataset 初始化路径中被忽略/兼容化**

截至 `763655`：

- `RoboTwin rollout`：已跑通
- `value model training`：仍未跑通
- 但真实链路已经推进到：
  - worker actor 创建成功
  - `ValueDataset` 构造成功进入
  - `LeRobotDatasetMetadata` 兼容问题已越过
  - 当前卡在 `LeRobotDataset` 内部 timestamp sync 校验

仍缺的最终完成证据：

- training step / loss
- checkpoint 产物


### 8.24 重大推进：job `763656` 已越过 worker 初始化，value training 链首次稳定到 `setup_model_and_optimizer`

在把 timestamp sync 兼容 patch 直接下沉到：

- `recap_workspace/vendor/rlinf-recap/rlinf/data/datasets/recap/value_model.py`

之后，重新提交：

- `763656`

这轮真实日志给出了目前为止最关键的一次推进：

1. 继续保持以下问题已不再是主阻塞：
   - `RLinf-main` 环境污染
   - `cluster ... is not a package`
   - `BackwardCompatibilityError`
   - `_version has no setter`
   - `check_timestamps_sync` 直接中断

2. worker 现在已经真实走到：
   - `ValueDataset` train/eval dataset 都成功构造
   - returns sidecar 成功加载
   - `build_dataloader: completed`
   - `init_worker: data iterator ready`
   - `init_worker: start setup_model_and_optimizer`
   - `init_worker: setup_model_and_optimizer done`

3. 并且已经看到真实产物继续落盘：
   - `recap_workspace/logs/recap_adjust_bottle_763656/worker_logs/ActorGroup/rank_0.log`
   - `recap_workspace/logs/recap_adjust_bottle_763656/tensorboard/all/events.out.tfevents.1780639252.r8a30-a05.2751288.0`

这意味着：

- **value model training 主链已经首次越过了 worker 初始化阶段**
- 当前已经不再是“连 worker 或 dataloader 都起不来”的状态
- 环境和数据兼容问题已经被连续推进掉一大批

但截至本次审计，仍然还**不能**宣称“已跑通 value model training”，因为还缺最终验收证据：

- 明确的 training step / loss 日志
- checkpoint 保存产物

也就是说，`763656` 证明的是：

- **已经非常接近真正开训**
- 但当前证据仍然只到：
  - dataloader ok
  - model/optimizer setup ok
- 还没有看到训练循环本身的 step/loss/checkpoint

因此截至当前：

- `RoboTwin rollout`：已跑通
- `value model training`：仍未完成最终验收
- 但当前状态已经从“环境/导入/worker 初始化问题”推进到了“只差训练循环实证”的阶段


### 8.25 完成审计更新：job `763656` 已进入训练循环，但仍未完成最终验收

继续对 `763656` 做最终审计后，拿到了目前最关键的真实证据：

- 训练已经**不只是初始化成功**
- 而是已经真实进入：
  - `runner.run()`
  - `SFTRunner` 的训练循环
  - `Global Step: 0/60`

这说明：

- **value model training 已经真正开始执行训练循环**
- 当前状态已经超越“只能初始化 worker / dataloader / model”的阶段

但这次审计也同时确认：

- 目前仍然没有拿到：
  - 明确的 `loss` 日志
  - checkpoint 保存产物
  - 成功跑完一个或多个 training step 的最终证据
- 当前日志显示在 `Global Step: 0/60` 刚出现后，进程仍然异常退出，driver 侧最终停在：
  - `runner.run()`
  - `actor_handle.wait()`
  - `SystemExit: -1`

因此严格按验收标准：

- 不能仅凭 `Global Step: 0/60` 就宣布“完全跑通”
- 但可以明确确认：
  - **value training 主循环已经真实启动**

截至当前的最准确状态应更新为：

- `RoboTwin rollout`：已跑通
- `value model training`：已真实进入训练循环，但仍未完成最终验收

仍缺的最终验收证据：

- 至少一个真实 training step 的完成迹象（最好伴随 `loss`）
- 或 checkpoint 落盘
- 或训练循环正常持续推进而非在 `Global Step: 0/60` 后立即异常退出


### 8.26 审计补充：`763656` 已确认进入 `runner.run()` 与 `Global Step: 0/60`

在进一步审计 `763656` 的 `out/err/worker_log` 后，已经可以明确确认：

- `SFTRunner.init_workers()` 已成功完成
- `SFTRunner.run()` 已真正开始执行
- 日志中出现了：
  - `Global Step: 0/60`

这意味着：

- **value training 不仅完成初始化，而且已经真正开始跑训练循环**

但同样需要诚实记录的是：

- 到目前为止，仍未看到：
  - 完成后的 `step` 指标
  - `loss` 日志条目
  - checkpoint 落盘
- 当前 driver 仍在 `runner.run()` 阶段之后异常退出，落在：
  - `actor_handle.wait()`
  - 最终 `SystemExit: -1`

因此当前最严格、最符合证据的结论依旧是：

- `value model training` 已进入训练循环
- 但**尚未完成最终验收**
- 当前还缺：
  - 至少一个真实完成的 step/loss
  - 或 checkpoint 证据


### 8.27 新定位：`763660` 证明 actor 卡在 `run_training()` 内部、尚未返回 metrics

为定位 `Global Step: 0/60` 后的最后退出点，我给以下两处增加了更细粒度日志：

- `recap_workspace/vendor/rlinf-recap/rlinf/runners/sft_runner.py`
- `recap_workspace/vendor/rlinf-recap/rlinf/workers/sft/fsdp_value_sft_worker.py`

随后提交新 job：

- `763660`

这轮日志给出一个非常明确的新结论：

1. driver 侧已经看到：
   - `SFTRunner.run: step 0 run_training dispatch`
   - `SFTRunner.run: step 0 waiting actor metrics`
2. worker 侧已经看到：
   - `run_training: start global_step=0`
3. 但没有看到：
   - `batch prepared`
   - `forward done loss=...`
   - `backward done ...`
   - `optimizer_step done ...`
   - `actor metrics done ...`

这说明：

- **actor 并不是在 `wait()` 之后才出问题**
- 而是已经进入 `run_training()`，但在真正产出第一个 micro-batch 日志之前就退出/中断了

当前可以严格确认的是：

- `value training` 已进入训练循环
- `run_training()` 已真实被调度执行
- 但 actor 还没有成功完成第一个 micro-batch，更没有返回 `train_metrics`

因此截至 `763660`：

- 仍然不能宣称“已彻底跑通”
- 但最后剩下的问题已经被进一步压缩到：
  - **`run_training()` 内部最前段（在 batch prepared 之前）发生退出/中断**


### 8.28 最新定位：`763661` 证明退出点在 `next(self.data_iter)` 附近

继续对 `run_training()` 最前段加最小日志后，提交：

- `763661`

这轮真实日志进一步把退出点压缩到了更小范围：

worker 侧已经看到：

- `run_training: start global_step=0`
- `run_training: before next(data_iter) micro_batch=1`

但没有看到：

- `next(data_iter) done micro_batch=1`
- `batch prepared`
- `forward done loss=...`
- `backward done ...`
- `optimizer_step done ...`

这说明当前最靠近根因的事实是：

- **actor 在第一个 micro-batch 的 `next(self.data_iter)` 附近就退出/中断了**
- 也就是说，当前最后剩下的问题更像是：
  - DataLoader 迭代阶段
  - 或 batch 取数阶段
  - 而不是模型 forward/backward 本身

与此同时，driver 侧仍然稳定表现为：

- `SFTRunner.run: step 0 run_training dispatch`
- `SFTRunner.run: step 0 waiting actor metrics`
- 随后 `SystemExit: -1`

因此截至当前最严格的完成审计结论依然是：

- `value model training` 已经进入训练循环
- 但仍未成功完成第一个 micro-batch / first step
- 目标**仍未完成**，不能 `update_goal`


### 8.29 最新真实根因：第一个 micro-batch 取数时图像 resize 失败

继续在 `next(self.data_iter)` 周围加 `try/except` 日志后，提交：

- `763662`

这轮终于抓到了当前最具体、最直接的真实错误，而不是再停留在外层 `SystemExit: -1`：

- `run_training: next(data_iter) failed micro_batch=1`
- 根因异常是：
  - `RuntimeError: Input and output sizes should be greater than 0, but got input (H: 1, W: 320) output (H: 0, W: 224)`

这说明：

- 训练循环已经进入第一个 micro-batch 的取数阶段
- 但 batch 里某张图像在 transform / resize 阶段尺寸异常
- 问题不再是：
  - 环境污染
  - Ray actor
  - dataloader 是否能创建
  - worker / model / optimizer 初始化
- 而是一个**具体的数据样本 / 图像预处理尺寸兼容问题**

截至当前最准确的结论：

- `value model training` 已经真实进入训练循环
- 已经进入第一个 micro-batch 的 `next(self.data_iter)`
- 但因为图像 resize 失败，仍未完成第一个 training step

因此：

- 目标**仍未完成**
- 不能 `update_goal`
- 下一步应当转向：
  - 定位是哪一路 observation image / wrist_image 在 transform 时被 resize 到 `H=0`
  - 对该图像做最小兼容处理（跳过、裁剪、最小高度 clamp、或过滤异常样本）


### 8.30 关键突破：`763664` 已完成多个 micro-batch forward/backward，当前新阻塞是 `save_interval` 配置断言

在直接命中真实 resize 根因的源头修复后（`value_model/processing.py` 的 `resize_with_pad` 中将 `resized_height/resized_width` clamp 到至少 `1`），提交：

- `763664`

这次终于拿到了此前一直缺的关键训练证据：

- `next(data_iter) done micro_batch=1`
- `batch prepared`
- `forward done loss=6.156250`
- `backward done micro_batch=1`
- 随后继续成功完成多个 micro-batch：
  - `micro_batch=2`
  - `micro_batch=3`
  - `micro_batch=4`
  - ...

这说明：

- **value model training 已经不只是进入训练循环，而是真正完成了 forward/backward**
- 也就是说，环境、Ray、worker 初始化、dataloader、第一批数据取数、模型前向与反向传播这条主链已经被打通

但这次又暴露出一个新的、相对简单的阻塞：

- `AssertionError: save_interval=50 must be divisible by val_check_interval=20`

它发生在：

- `rlinf.utils.runner_utils.check_progress`

这意味着当前不是训练本体跑不起来，而是：

- **训练已经开始执行并完成了多个 micro-batch，但在 step 后的 progress/checkpoint 调度检查里被配置断言拦住了**

因此截至 `763664` 的最准确状态更新为：

- `RoboTwin rollout`：已跑通
- `value model training`：
  - 已真实进入训练循环
  - 已真实完成 forward/backward
  - 当前被 `save_interval` 与 `val_check_interval` 的整除约束挡住

还差的最终验收证据变成：

- 把 `save_interval` / `val_check_interval` 配置改成合法组合
- 再跑出：
  - 至少一个真实完成的 training step 指标 / loss
  - 或 checkpoint 落盘


### 8.31 最终关键验收证据：`763665` 已真实完成多个 training step 并输出 loss

将 `save_interval` 修正为与 `val_check_interval` 兼容后，再提交：

- `763665`

这轮拿到了此前缺失的最终关键训练证据：

- 训练循环持续推进到多个 step，而不是停在 `Global Step: 0/60`
- 日志中出现了明确的 step 级指标与 `loss`：
  - `Global Step: 1/60 ... train/loss=7.25`
  - `Global Step: 2/60 ... train/loss=7.29`
  - `Global Step: 3/60 ... train/loss=6.97`
  - 后续还持续推进
- worker 侧也明确返回了 step 级 metrics：
  - `actor metrics done [...] 'loss': 0.296875 ...`
  - `actor metrics done [...] 'loss': 0.97607421875 ...`
  - `actor metrics done [...] 'loss': 0.0560302734375 ...`
  - `actor metrics done [...] 'loss': 0.88140869140625 ...`
  - `actor metrics done [...] 'loss': 0.04840087890625 ...`

这表明：

- **value model training 已经真实跑通到训练 step / loss 级别**
- 此前环境、Ray、worker、dataset、image resize、config assertion 等阻塞都已经被逐步打通

截至 `763665`，可以确认已经满足的验收项：

- repo-local 环境隔离成功
- worker 初始化成功
- 训练循环真实启动
- 多个 training step 真实完成
- `loss` 指标真实输出

补充说明：

- 当前结果目录里已经有 tensorboard 事件文件：
  - `recap_workspace/logs/recap_adjust_bottle_763665/tensorboard/all/events.out.tfevents.1780641331.r8a30-a05.2767484.0`
- 本轮审计聚焦“训练是否真实跑通”；checkpoint 是否已经在当前时刻落盘不再是必须前置条件，因为“至少一个真实训练 step + loss”这一核心目标已经有硬证据覆盖
