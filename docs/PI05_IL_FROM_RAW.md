# PI05 原始数据 IL 训练说明

这条链路面向目录：

- `/nfs_global/S/yangrongzheng/pick_place_raw_data`

目标是**不依赖当前 RL 训练脚本**，直接把 `pkl` 原始轨迹拿来做 imitation learning。

## 1. 现在可直接使用的入口

### 直接从原始 PKL 训练 IL baseline

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
python scripts/train_il_from_pkl.py --max-episodes 8 --epochs 5 --batch-size 32
```

输出目录默认在：

- `/nfs_global/S/yangrongzheng/pi05/results/pi05_il_raw`

会生成：

- `best.pt`
- `last.pt`

## 2. 当前 action 定义

默认使用：

- `joint_and_gripper_delta`

也就是：

- 前 6 维来自 `episode_ur5e_pos_j` 的相邻帧差分
- 最后 1 维来自 `episode_inspire_hand_pos[:, :1]` 的相邻帧差分

可选：

```bash
python scripts/train_il_from_pkl.py --action-mode joint_delta
python scripts/train_il_from_pkl.py --action-mode eef_delta
```

## 3. 当前 state 定义

训练输入 state 由三部分拼接：

- `episode_ur5e_pos_j` (6)
- `episode_ur5e_pos_eef` (6)
- `episode_inspire_hand_pos` (6)

总维度：

- `18`

## 4. 当前模型

当前为了先把数据和训练链跑通，使用的是一个 MLP policy baseline：

- 输入：18 维状态
- 输出：action 向量
- 损失：MSE

这意味着：

- **现在已经能直接训练 IL**
- 但**还不是 openpi/pi05 的视觉-语言动作模型微调**

## 5. LeRobot 转换脚本

仓库里还提供了一个转换脚本雏形：

```bash
python scripts/convert_pick_place_to_lerobot.py --max-episodes 10 --overwrite
```

输出目录默认：

- `/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place`

这一步目前还在继续打磨，目的是后续接到 openpi / LeRobot 的标准离线训练链路。

## 6. 建议使用顺序

如果你现在的目标是“先直接跑通 IL”，建议顺序：

```bash
python scripts/train_il_from_pkl.py --max-episodes 2 --epochs 1 --batch-size 16
python scripts/train_il_from_pkl.py --max-episodes 8 --epochs 5 --batch-size 32
```

先小样本 smoke，再扩大数据。

## 7. 文件入口

- `scripts/train_il_from_pkl.py`
- `scripts/convert_pick_place_to_lerobot.py`
