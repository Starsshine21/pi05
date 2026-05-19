# PI05 真机推理接入说明

本说明描述当前目录下新增的 PI05 真机推理脚本如何连接：

- UR5e 机械臂
- Inspire hand
- Orbbec Femto Bolt 相机
- L515 相机

并加载 `openpi_official` 训练得到的 checkpoint 进行在线推理。

## 1. 新增脚本

- `scripts/pi05_real_robot_infer.py`

## 2. 参考来源

该脚本参考了 `newRL` 的真机 API 细节（RTDE / serial / camera SDK），但动作解释和推理逻辑按当前 PI05 训练链定义实现。

参考文件：

- `/nfs_global/S/yangrongzheng/newRL/3D-Diffusion-Policy/diffusion_policy_3d/env/real_robot/ur5e_inspire_dualcam_env.py`

## 3. 当前推理输入

脚本采集并组织为以下 observation：

- `image`: 来自 `L515`
- `wrist_image`: 来自 `Orbbec Femto Bolt`
- `state`: `concat([joints, eef, hand])` 共 18 维
- `prompt`: 用户传入的任务文本

与当前训练数据 schema 保持一致：

- `image`
- `wrist_image`
- `state`
- `prompt`

## 4. 当前推理输出解释

当前脚本按现有训练数据定义解释模型输出：

- 输出前 12 维动作
- 前 6 维：`joint delta`
- 后 6 维：`hand delta`

即：

- `target_joints = current_joints + action[:6]`
- `target_hand = current_hand + action[6:12]`

然后：

- joint 通过 UR5e RTDE `servoJ` 下发
- hand 通过 Inspire hand serial `set_hand_pos` 下发

## 5. 注意事项

当前实现是“最小可用版”：

- 默认使用 `pi05_pickplace_full_pytorch` checkpoint
- 默认按 joint-delta 解释模型输出
- 没有加入额外安全边界、碰撞检测、任务级状态机
- 真机运行前请务必先在安全环境中小步验证

## 6. 启动示例

```bash
source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh
conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final
source /nfs_global/S/yangrongzheng/pi05/scripts/use_local_openpi_env.sh

PYTHONPATH=/nfs_global/S/yangrongzheng/pi05/openpi_official/src:$PYTHONPATH \
python scripts/pi05_real_robot_infer.py \
  --checkpoint-dir /nfs_global/S/yangrongzheng/pi05/results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch/<EXP_NAME>/60000 \
  --train-config pi05_pickplace_full_pytorch \
  --prompt "place the cookie at the left top position on the desk" \
  --robot-ip 192.168.1.109 \
  --hand-port /dev/ttyUSB0
```

## 7. 当前与 newRL 的差异

虽然参考了 `newRL` 的硬件 API，但当前脚本**没有沿用 newRL 的动作语义**：

- `newRL` 真机链更偏 `next-state` / normalized target
- 当前脚本按当前 PI05 训练数据定义，使用 `joint + hand delta`

这是有意为之，因为当前推理应与当前训练数据保持一致。
