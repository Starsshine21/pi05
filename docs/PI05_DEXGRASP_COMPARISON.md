# PI05 与 DexGraspVLA 链路梳理

## 1. 直观概念

### 1.1 joints
- `joints` 表示机械臂胳膊各个关节本身的角度。
- 对 UR5e 来说通常是 6 维：6 个关节角。
- 直观理解：肩膀、肘子、手腕各自弯了多少。

### 1.2 eef
- `eef` 表示末端执行器（end-effector）的空间位姿。
- 通常是 6 维：`x, y, z, rx, ry, rz`。
- 前 3 维是位置，后 3 维是姿态旋转向量（rotvec）。
- 直观理解：手整体在空间里哪里、朝哪个方向。

### 1.3 hand
- `hand` 表示灵巧手的控制状态。
- 当前链路里是 6 维，不是完整 21 个物理 DOF。
- 这 6 维是 Inspire hand 驱动接口暴露的 6 个控制通道。
- 直观理解：小拇指/无名指/中指/食指/拇指弯曲/拇指旋转各自收了多少。

## 2. 当前 PI05 链路是怎么定义的

### 2.1 状态输入
在 `scripts/convert_pick_place_to_lerobot.py` 中：
- `state = concat([joints, eef, hand])`
- 共 18 维。

即：
- 前 6 维：UR5e joint
- 中间 6 维：EEF pose
- 后 6 维：Inspire hand pos

### 2.2 动作输出
当前主链使用 `eef_delta`：
- `base = concat([eef, hand])`
- `action[t] = base[t+1] - base[t]`

即：
- 前 6 维：`eef_delta`
- 后 6 维：`hand_delta`

### 2.3 真机执行
在 `pi05_real_robot_infer.py` 中：
- `action[:6]` 当作 `eef_delta`
- `action[6:12]` 当作 `hand_delta`
- `target_eef = current_eef + eef_delta`
- 再通过 IK 求 `target_joints`
- `target_hand = current_hand + hand_delta`
- 再写回 `set_hand_pos()`

## 3. DexGraspVLA 是怎么定义的

## 3.1 数据处理（`../dex-data-scripts/pkl_to_zarr.py`）
DexGraspVLA 的 zarr 构造方式是：
- `right_state = concat([normalize(episode_ur5e_pos_j), normalize(episode_inspire_hand_pos)])`
- 不使用 `eef`

即训练输入状态更接近：
- `joint + hand`

### 3.2 动作构造
- `episode_action = episode_right_state[1:]`（末帧复制最后一帧）

即动作本质上是：
- **下一时刻的 joint+hand 目标状态**
- 不是 delta
- 不是 eef

### 3.3 推理执行（`../DexGraspVLA-yx/inference.py`）
- 前一部分动作直接缩放到 arm joint 范围并执行
- 后一部分动作直接交给 hand 执行接口

也就是说 DexGraspVLA 的控制方式更接近：
- **joint target + hand target**

## 4. hand 映射是否正确

### 4.1 DexGraspVLA hand 顺序
在 `../DexGraspVLA-yx/hardware_handler/inspire_hand.py`：
- `0 = LITTLE`
- `1 = RING`
- `2 = MIDDLE`
- `3 = INDEX`
- `4 = THUMB_BEND`
- `5 = THUMB_ROTATE`

底层按 `0..5` 顺序读写寄存器，没有重排。

### 4.2 PI05 hand 顺序
在 `pi05_real_robot_infer.py`：
- `get_hand_pos()` 按 `0..5` 顺序读 `posAct`
- `set_hand_pos()` 按 `0..5` 顺序写 `posSet`
- 寄存器地址与 DexGraspVLA 一致：
  - `posSet = 1474`
  - `posAct = 1534`

### 4.3 结论
- 从代码链路上看，当前没有看到 hand 6 维顺序错位的证据。
- 也就是说，数据采集、DexGraspVLA、当前 PI05 推理三边都在使用同一套 6 维 hand 通道顺序。
- 但若要 100% 封口，仍建议做一次真机“单维激励”测试。

## 5. IK 是否无偏差

### 5.1 不能认为无偏差
即便 UR 的 IK 算法本身没 bug，也不能假设：
- 训练数据里的 `eef` 与真机实时 `eef` 定义完全一致
- 同一个 TCP pose 总能回到和采集时完全相同的 joint 解
- 小的 `eef_delta` 不会引起 joint 解切换

### 5.2 当前判断
- IK 可能带来次级误差
- 但从当前 MSE 结果看，`eef` 误差并不大
- 当前主问题不像是 IK，而更像是 hand 的建模方式

## 6. 当前最可能的问题所在

### 6.1 不是 hand 顺序错位
- 目前没有看到明确证据。

### 6.2 更像是动作空间选得不合适
- DexGraspVLA 成功链路：`joint + hand target`
- 当前 PI05 链路：`eef_delta + hand_delta`

也就是说：
- 原始数据本来更像支持 joint-space target 学习
- 当前 PI05 重新解释成 eef-space delta + hand-space delta
- 这会增加学习难度，尤其 hand 更难学

### 6.3 hand 是当前主瓶颈
从 2-episode MSE 看：
- `eef` 维度误差已经很小
- `hand` 6 维误差巨大，主导总 MSE

## 7. 现阶段建议

### 优先级 1
先不要继续死磕 `hand_delta`，尝试把 hand 改成 target：
- 前 6 维继续 `eef_delta`
- 后 6 维改成 `hand_target`

### 优先级 2
更进一步，尝试完全对齐 DexGraspVLA 的动作空间：
- `state = joint + hand`
- `action = next_state(joint + hand)`

### 优先级 3
最后再考虑：
- 手部降维
- 更复杂的 IK / scale / eef rotvec 微调

## 8. 一句话总结

- 当前没有发现 hand 6 维映射错误的证据。
- 当前最大问题更像是：
  - **你把原始成功数据的动作空间，从 `joint+hand target` 改成了 `eef_delta + hand_delta`，这很可能不适合这批数据，尤其不适合 hand。**
