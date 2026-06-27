# 新增两条 PI05 数据/推理链路

## 1. `eef + hand target`

### 训练数据
- `state-mode=eef_hand`
- `action-mode=eef_delta --use-next-state-action`
  - 这里会生成 `action = [eef[t+1], hand[t+1]]`
  - 因为 `eef_delta` 的 base 是 `concat([eef, hand])`

示例：
```bash
python scripts/convert_pick_place_to_lerobot.py \
  --output-dir data/lerobot_pick_place_eef_hand_target \
  --state-mode eef_hand \
  --action-mode eef_delta \
  --use-next-state-action \
  --overwrite
```

### 真机推理
- `--state-mode eef_hand`
- `--action-mode eef_hand_target`

## 2. `joint + hand target`

### 训练数据
- `state-mode=joint_hand`
- `action-mode=joint_and_gripper_delta --use-next-state-action`
  - 这里会生成 `action = [joints[t+1], hand[t+1]]`

示例：
```bash
python scripts/convert_pick_place_to_lerobot.py \
  --output-dir data/lerobot_pick_place_joint_hand_target \
  --state-mode joint_hand \
  --action-mode joint_and_gripper_delta \
  --use-next-state-action \
  --overwrite
```

### 真机推理
- `--state-mode joint_hand`
- `--action-mode joint_hand_target`

## 3. 说明
- 当前改动只接通了数据转换与真机推理的状态/动作语义。
- 如果训练配置对 `state_dim` 或 `action_dim` 有硬编码假设，需要确保训练时使用的数据 schema 与 checkpoint/config 一致。
- 现有 `pi05_pickplace_full_pytorch` 的模型输出前 12 维仍可被上述两条链路消费。
