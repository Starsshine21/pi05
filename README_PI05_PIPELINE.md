# PI05 PickPlace 数据与训练链路说明

本文档说明当前仓库中，pick-place 真机原始数据如何被转换为 LeRobot 数据集，以及这份数据如何接入 `openpi_official/` 下的训练配置。

## 1. 原始数据位置与格式

原始数据目录：

- `/nfs_global/S/yangrongzheng/pick_place_raw_data`

每个 episode 是一个 `.pkl` 文件，例如：

- `0_light-on_desk_cookie_left_top.pkl`

原始字段包含：

- `episode_ur5e_pos_j`: 机械臂 joint 状态，形状 `(T, 6)`
- `episode_ur5e_pos_eef`: 末端执行器状态，形状 `(T, 6)`
- `episode_inspire_hand_pos`: inspire hand 状态，形状 `(T, 6)`
- `episode_l515_color`: 一路 RGB 图像，形状 `(T, 480, 640, 3)`
- `episode_orbbec_femto_bolt_color`: 一路 RGB 图像，形状 `(T, 720, 1280, 3)`

文件名同时编码了任务语义，例如：

- `0_light-on_desk_cookie_left_top.pkl`

会被解析成：

- object = `cookie`
- surface = `desk`
- target = `left_top`

并转换为文本指令：

- `place the cookie at the left top position on the desk`

## 2. 当前 LeRobot 转换脚本

转换脚本：

- `scripts/convert_pick_place_to_lerobot.py`

当前脚本输出为 LeRobot dataset，核心字段如下：

- `image`
- `wrist_image`
- `state`
- `actions`
- `prompt`

### 2.1 图像映射

当前映射为：

- `image` <- `episode_l515_color`
- `wrist_image` <- `episode_orbbec_femto_bolt_color`

两路图像在转换时统一 resize 到 `224 x 224`。

### 2.2 state 定义

当前 `state` 定义为：

- `state = concat([joints, eef, hand])`

因此：

- `state_dim = 6 + 6 + 6 = 18`

也就是说，模型看到的低维观测包含：

- 机械臂 joint 状态
- 末端执行器状态
- 手部状态

### 2.3 actions 定义

当前默认动作模式为：

- `joint_and_gripper_delta`

对应定义：

- `base = concat([joints, hand])`
- `actions[t] = base[t+1] - base[t]`

因此：

- `action_dim = 6 + 6 = 12`

也就是：

- 前 6 维：机械臂 joint delta
- 后 6 维：hand delta

脚本还支持：

- `--use-next-state-action`

如果开启，则：

- `actions[t] = base[t+1]`

即动作被定义为下一时刻目标状态，而不是 delta。

当前这次重建数据时，使用的是默认 delta 形式。

## 3. 转换后的 LeRobot 数据集

输出目录：

- `data/lerobot_pick_place`

当前 `meta/info.json` 中的 schema 为：

- `image`: `[224, 224, 3]`
- `wrist_image`: `[224, 224, 3]`
- `state`: `[18]`
- `actions`: `[12]`
- `prompt`: string

转换结果：

- 成功 episode 数：`496`
- 跳过损坏 pkl：`44`

## 4. 为什么不用裸 `pi05_libero`

虽然当前转换后的字段名已经尽量向官方 `LeRobotLiberoDataConfig` 对齐，但数据语义仍然是当前真机 pick-place 自己的定义：

- `state = 18`
- `actions = 12`

而不是标准 LIBERO 示例中的状态/动作定义。

因此，训练不直接使用 `openpi/` 下原生 `pi05_libero`，而是使用 `openpi_official/` 下已经适配这份数据的 pickplace config。

## 5. 当前训练配置

训练仓库：

- `openpi_official/`

当前主要配置：

### 5.1 JAX LoRA

配置名：

- `pi05_pickplace_lora`

位置：

- `openpi_official/src/openpi/training/config.py`
- `openpi_official/src/openpi/training/pi05_pickplace_config.py`

关键点：

- `action_dim = 12`
- `action_horizon = 10`
- schema 映射：
  - `image -> observation/image`
  - `wrist_image -> observation/wrist_image`
  - `state -> observation/state`
  - `actions -> actions`
  - `prompt -> prompt`

### 5.2 PyTorch Full

配置名：

- `pi05_pickplace_full_pytorch`

位置：

- `openpi_official/src/openpi/training/config.py`

关键点：

- 模型内部 `action_dim = 32`
- 数据环境动作通过 `PickPlaceOutputs` 截取前 `12` 维
- 当前数据 schema 同样映射为：
  - `image`
  - `wrist_image`
  - `state`
  - `actions`
  - `prompt`

这意味着：

- 数据真实动作维度是 `12`
- 模型内部 action 通道仍然保持 PI05 所需的 `32`
- 训练与推理时通过 transform / output adapter 对齐

## 6. norm stats 与训练入口

训练前必须先计算 norm stats。

当前 PyTorch Full 的一体化 slurm：

- `scripts/pi05_official_pickplace_pytorch_full.slurm`

这个脚本会自动：

1. 检查并挂好 LeRobot 数据软链
2. 运行 `compute_norm_stats.py`
3. 运行 `train_pytorch.py`

默认使用：

- norm stats config: `pi05_pickplace_lora`
- train config: `pi05_pickplace_full_pytorch`

## 7. 使用方法

提交 PyTorch Full 训练：

```bash
sbatch scripts/pi05_official_pickplace_pytorch_full.slurm
```

日志输出位置：

- `logs/pi05-pickplace-pt-full-<jobid>.out`
- `logs/pi05-pickplace-pt-full-<jobid>.err`

## 8. 设计总结

当前链路的设计原则是：

- raw data 保留真机原始语义
- LeRobot 层尽量对齐官方 LIBERO 风格字段名
- 训练层在 `openpi_official/` 下保留 pickplace 专用 config
- 不修改 `openpi/` 原仓库

这样既能尽量复用官方训练链路，也能保持当前真机数据的状态与动作定义不被错误压缩。
