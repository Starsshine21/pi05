# PI05 两条新链路完整流程说明

本文档说明两条新链路如何从 **raw pkl 数据** 开始，经过 **数据转换**、**norm stats 计算**、**训练**，最后对应到 **推理执行语义**。

---

# 1. 背景

当前新增两条链路：

1. **`eef + hand target`**
2. **`joint + hand target`**

它们和原来的 `joint + eef + hand` 输入、`eef_delta + hand_delta` 输出不同。

## 原链路
- 输入 state：`joint + eef + hand`（18 维）
- 输出 action：`eef_delta + hand_delta`（12 维）

## 新链路 A：`eef + hand target`
- 输入 state：`eef + hand`（12 维）
- 输出 action：`next_eef + next_hand`（12 维）

## 新链路 B：`joint + hand target`
- 输入 state：`joint + hand`（12 维）
- 输出 action：`next_joint + next_hand`（12 维）

---

# 2. 原始数据来源

原始数据是 pick-and-place 采集得到的 pkl 文件：

- 输入目录：`/nfs_global/S/yangrongzheng/pick_place_raw_data`

每个 pkl 中至少包含这些字段：

- `episode_ur5e_pos_j`
- `episode_ur5e_pos_eef`
- `episode_inspire_hand_pos`
- `episode_l515_color`
- `episode_orbbec_femto_bolt_color`

这些字段在 `scripts/convert_pick_place_to_lerobot.py` 中被读取并转换。

---

# 3. 负责数据转换的脚本

统一使用：

- `scripts/convert_pick_place_to_lerobot.py`

这个脚本现在支持：

## `--state-mode`
可选：
- `joint_eef_hand`
- `joint_hand`
- `eef_hand`

## `--action-mode`
可选：
- `joint_delta`
- `eef_delta`
- `joint_and_gripper_delta`

## `--use-next-state-action`
如果打开：
- 不再构造 delta
- 而是直接使用下一时刻目标状态作为动作

---

# 4. 链路 A：`eef + hand target`

## 4.1 训练输入是什么
输入 state 只保留：
- `eef`
- `hand`

对应参数：
- `--state-mode eef_hand`

这会把：
- `state = concat([eef, hand])`

维度为：
- `6 + 6 = 12`

---

## 4.2 训练目标是什么
目标 action 是：
- `next_eef + next_hand`

对应做法：
- `--action-mode eef_delta`
- `--use-next-state-action`

原因：
- `eef_delta` 这条 base 本来就是 `concat([eef, hand])`
- 开启 `--use-next-state-action` 后，脚本会直接取下一时刻 base

所以最终动作就是：
- `action[t] = [eef[t+1], hand[t+1]]`

维度为：
- `6 + 6 = 12`

---

## 4.3 数据转换命令
```bash
python scripts/convert_pick_place_to_lerobot.py \
  --input-dir /nfs_global/S/yangrongzheng/pick_place_raw_data \
  --output-dir /nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_eef_hand_target \
  --repo-id local/pi05-pickplace-eef-hand-target \
  --state-mode eef_hand \
  --action-mode eef_delta \
  --use-next-state-action \
  --overwrite \
  --stride 1 \
  --image-height 224 \
  --image-width 224
```

输出数据目录：
- `data/lerobot_pick_place_eef_hand_target`

repo_id：
- `local/pi05-pickplace-eef-hand-target`

---

## 4.4 训练 config
使用新增 config：
- `pi05_pickplace_full_pytorch_eef_hand_target`

定义位置：
- `openpi_official/src/openpi/training/config.py`

它对应：
- `repo_id = local/pi05-pickplace-eef-hand-target`
- `asset_id = local/pi05-pickplace-eef-hand-target`

---

## 4.5 norm stats
统一使用：
- `openpi_official/scripts/compute_norm_stats.py`

命令：
```bash
cd openpi_official
python scripts/compute_norm_stats.py \
  pi05_pickplace_full_pytorch_eef_hand_target \
  --assets-base-dir /nfs_global/S/yangrongzheng/pi05/openpi_official/assets_eef_hand_target
```

norm stats 输出位置：
- `openpi_official/assets_eef_hand_target/local/pi05-pickplace-eef-hand-target/norm_stats.json`

---

## 4.6 正式训练
正式 slurm：
- `scripts/pi05_eef_hand_target_full.slurm`

它会自动做三件事：
1. 从 raw pkl 转数据
2. 计算 norm stats
3. 启动训练

训练 config：
- `pi05_pickplace_full_pytorch_eef_hand_target`

训练结果 checkpoint 根目录：
- `results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch`

实验名：
- `pi05_pickplace_eef_hand_target_train`

---

## 4.7 推理语义
真机推理时对应：
- `--state-mode eef_hand`
- `--action-mode eef_hand_target`

含义：
- 模型输出前 6 维直接视作 `target_eef`
- 模型输出后 6 维直接视作 `target_hand`
- `target_eef` 仍需 IK 求 `target_joints`
- `target_hand` 直接写入手

所以这条链路是：
- **输入：当前 `eef + hand`**
- **输出：下一时刻 `eef + hand` 目标**

---

# 5. 链路 B：`joint + hand target`

## 5.1 训练输入是什么
输入 state 只保留：
- `joint`
- `hand`

对应参数：
- `--state-mode joint_hand`

这会把：
- `state = concat([joints, hand])`

维度为：
- `6 + 6 = 12`

---

## 5.2 训练目标是什么
目标 action 是：
- `next_joint + next_hand`

对应做法：
- `--action-mode joint_and_gripper_delta`
- `--use-next-state-action`

原因：
- `joint_and_gripper_delta` 的 base 是 `concat([joints, hand])`
- 打开 `--use-next-state-action` 后，直接取下一时刻 base

所以：
- `action[t] = [joint[t+1], hand[t+1]]`

维度为：
- `6 + 6 = 12`

---

## 5.3 数据转换命令
```bash
python scripts/convert_pick_place_to_lerobot.py \
  --input-dir /nfs_global/S/yangrongzheng/pick_place_raw_data \
  --output-dir /nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place_joint_hand_target \
  --repo-id local/pi05-pickplace-joint-hand-target \
  --state-mode joint_hand \
  --action-mode joint_and_gripper_delta \
  --use-next-state-action \
  --overwrite \
  --stride 1 \
  --image-height 224 \
  --image-width 224
```

输出数据目录：
- `data/lerobot_pick_place_joint_hand_target`

repo_id：
- `local/pi05-pickplace-joint-hand-target`

---

## 5.4 训练 config
使用新增 config：
- `pi05_pickplace_full_pytorch_joint_hand_target`

定义位置：
- `openpi_official/src/openpi/training/config.py`

它对应：
- `repo_id = local/pi05-pickplace-joint-hand-target`
- `asset_id = local/pi05-pickplace-joint-hand-target`

---

## 5.5 norm stats
统一使用：
- `openpi_official/scripts/compute_norm_stats.py`

命令：
```bash
cd openpi_official
python scripts/compute_norm_stats.py \
  pi05_pickplace_full_pytorch_joint_hand_target \
  --assets-base-dir /nfs_global/S/yangrongzheng/pi05/openpi_official/assets_joint_hand_target
```

norm stats 输出位置：
- `openpi_official/assets_joint_hand_target/local/pi05-pickplace-joint-hand-target/norm_stats.json`

---

## 5.6 正式训练
正式 slurm：
- `scripts/pi05_joint_hand_target_full.slurm`

它会自动做三件事：
1. 从 raw pkl 转数据
2. 计算 norm stats
3. 启动训练

训练 config：
- `pi05_pickplace_full_pytorch_joint_hand_target`

训练结果 checkpoint 根目录：
- `results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch`

实验名：
- `pi05_pickplace_joint_hand_target_train`

---

## 5.7 推理语义
真机推理时对应：
- `--state-mode joint_hand`
- `--action-mode joint_hand_target`

含义：
- 模型输出前 6 维直接视作 `target_joints`
- 模型输出后 6 维直接视作 `target_hand`
- 不经过 `eef_delta`
- 不需要先构造目标 TCP 再 IK

所以这条链路是：
- **输入：当前 `joint + hand`**
- **输出：下一时刻 `joint + hand` 目标**

---

# 6. 两条链路各自用什么

## 链路 A：`eef + hand target`
- 原始输入：`pick_place_raw_data/*.pkl`
- 转换脚本：`scripts/convert_pick_place_to_lerobot.py`
- 转换参数：
  - `--state-mode eef_hand`
  - `--action-mode eef_delta`
  - `--use-next-state-action`
- repo_id：`local/pi05-pickplace-eef-hand-target`
- train config：`pi05_pickplace_full_pytorch_eef_hand_target`
- norm stats 脚本：`openpi_official/scripts/compute_norm_stats.py`
- 正式训练 slurm：`scripts/pi05_eef_hand_target_full.slurm`
- 推理模式：
  - `--state-mode eef_hand`
  - `--action-mode eef_hand_target`

## 链路 B：`joint + hand target`
- 原始输入：`pick_place_raw_data/*.pkl`
- 转换脚本：`scripts/convert_pick_place_to_lerobot.py`
- 转换参数：
  - `--state-mode joint_hand`
  - `--action-mode joint_and_gripper_delta`
  - `--use-next-state-action`
- repo_id：`local/pi05-pickplace-joint-hand-target`
- train config：`pi05_pickplace_full_pytorch_joint_hand_target`
- norm stats 脚本：`openpi_official/scripts/compute_norm_stats.py`
- 正式训练 slurm：`scripts/pi05_joint_hand_target_full.slurm`
- 推理模式：
  - `--state-mode joint_hand`
  - `--action-mode joint_hand_target`

---

# 7. 最后一句话总结

如果你想从 raw pkl 开始完整跑通：

- `eef + hand target`：
  - 用 `scripts/pi05_eef_hand_target_full.slurm`
- `joint + hand target`：
  - 用 `scripts/pi05_joint_hand_target_full.slurm`

这两个 slurm 都会从：
- **raw pkl → LeRobot 数据 → norm stats → 训练**
一条链直接跑到底。
