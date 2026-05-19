# PI05 / RECAP 使用说明

  这份 `README.md` 只保留三部分你最关心的内容：

  1. 如何复现环境
  2. 如何跑 PI05 的真机推理
  3. 如何跑 RECAP 的真机强化学习

  默认前提：

  - 你会自己准备好 `pi05` 的 policy model checkpoint
  - 你会自己准备好 RECAP 所需的 value model
  - 本仓库只负责代码、环境、脚本和流程，不包含日志、训练数据和大模型文件

  ---

  ## 1. 如何复现环境

  这一部分的目标是：

  - 在一台新机器上把当前仓库的运行环境搭起来
  - 能成功 import `openpi_official`
  - 能加载 PI05 checkpoint
  - 能进一步跑真机推理或 RECAP 真机采集/训练

  ### 1.1 推荐路线

  当前仓库支持两条路线：

  - 路线 A：继续依赖 `RLinf-main`
  - 路线 B：不依赖 `RLinf-main`，直接在目标机重建独立环境

  如果你现在是在一台“什么都没有”的新机器上复现，**推荐优先走路线 B**。

  ---

  ### 1.2 路线 A：兼容当前老环境（依赖 `RLinf-main`）

  这条路线最贴近当前机器的运行方式。

  #### 1.2.1 需要的东西

  - 当前仓库代码
  - 一份可用的 `RLinf-main`
  - Python 3.11 conda 环境
  - 能用的 CUDA / torch / jax 环境

  #### 1.2.2 当前环境脚本

  老环境用的是：

  ```bash
  scripts/use_local_openpi_env.sh

  这个脚本依赖：

  - RLinf-main/.venv/lib/python3.11/site-packages
  - RLinf-main/RLinf_deps/libero
  - 对应 CUDA runtime 路径

  #### 1.2.3 使用方式

  source /path/to/miniconda3/etc/profile.d/conda.sh
  conda activate /path/to/pi05/.conda-pi05-openpi-final

  export PI05_CONDA_ENV_ROOT=/path/to/pi05/.conda-pi05-openpi-final
  export RLINF_ROOT=/path/to/RLinf-main
  export RLINF_VENV_ROOT=/path/to/RLinf-main/.venv
  export RLINF_VENV_SITE_PACKAGES=/path/to/RLinf-main/.venv/lib/python3.11/site-packages
  export LIBERO_REPO_PATH=/path/to/RLinf-main/RLinf_deps/libero

  source scripts/use_local_openpi_env.sh

  如果你只是想尽快复刻当前机器行为，这条路线最省事。

  ———

  ### 1.3 路线 B：独立环境复现（推荐）

  这条路线不依赖 RLinf-main，更适合新机器部署。

  #### 1.3.1 目标机最少需要什么

  - Linux
  - Python 3.11
  - conda / mamba
  - NVIDIA 驱动（如果要 GPU 推理）
  - git

  #### 1.3.2 拉代码

  git clone https://github.com/Starsshine21/pi05.git
  cd pi05

  #### 1.3.3 创建独立 conda 环境

  conda create -n pi05 python=3.11 -y
  conda activate pi05

  #### 1.3.4 安装 openpi_official

  cd openpi_official
  pip install -e .
  cd ..

  #### 1.3.5 安装仓库补充依赖

  pip install -r requirements-standalone.txt

  目前这个文件中最小补充包是：

  - opencv-python
  - pyserial

  #### 1.3.6 启动独立环境脚本

  独立环境专用脚本是：

  scripts/use_local_openpi_env_standalone.sh

  使用方式：

  source /path/to/miniconda3/etc/profile.d/conda.sh
  conda activate pi05
  source scripts/use_local_openpi_env_standalone.sh

  这个脚本会：

  - 设置 PYTHONPATH 到 openpi_official/src
  - 设置 HF_HOME / HF_DATASETS_CACHE / OPENPI_DATA_HOME
  - 如果存在 data/lerobot_pick_place，尝试准备本地 lerobot 入口

  #### 1.3.7 检查环境是否成功

  环境检查脚本：

  scripts/check_standalone_openpi_env.py

  执行：

  python scripts/check_standalone_openpi_env.py

  它会检查：

  - cv2
  - numpy
  - torch
  - serial
  - openpi.training.config
  - openpi.policies.policy_config

  还会顺手检查真机相关依赖是否已安装：

  - pyrealsense2
  - pyorbbecsdk
  - rtde_control
  - rtde_receive

  ———

  ### 1.4 真机相关额外依赖

  如果你只做离线加载 checkpoint，不一定需要真机依赖。

  如果你要跑 PI05 真机推理或 RECAP 真机采集，目标机还需要：

  - rtde_control
  - rtde_receive
  - pyserial
  - pyrealsense2
  - pyorbbecsdk

  对应硬件含义：

  - rtde_control / rtde_receive：UR5e 机械臂
  - pyserial：Inspire hand
  - pyrealsense2：L515
  - pyorbbecsdk：Orbbec Femto Bolt

  ———

  ### 1.5 你自己需要准备的模型

  这个仓库不包含模型文件，所以你需要自己额外准备：

  #### PI05 policy model

  你需要准备一个可推理的 PI05 checkpoint 目录，例如：

  results/openpi_official_pytorch_full_checkpoints/pi05_pickplace_full_pytorch/pi05_pickplace_full_pytorch_757027/60000

  推理至少需要：

  - model.safetensors
  - metadata.pt
  - assets/local/pi05-pickplace-il/norm_stats.json

  #### RECAP value model

  你会自己下载 RECAP 所需的 value model。

  本仓库只假设：

  - 你已经有可用的 value model 路径
  - 你会在 RECAP 运行命令中显式指定它

  ———

  ## 2. 如何跑 PI05 的真机推理

  这一部分只讲最小可用的 PI05 真机推理链路。

  ### 2.1 入口脚本

  PI05 真机推理脚本是：

  scripts/pi05_real_robot_infer.py

  它会完成：

  - 连接 UR5e
  - 连接 Inspire hand
  - 打开 L515
  - 打开 Orbbec Femto Bolt
  - 读取 observation
  - 加载 openpi_official policy
  - 输出 action
  - 下发到机械臂和手爪

  ### 2.2 当前 observation 定义

  推理输入是：

  - image：来自 L515
  - wrist_image：来自 Orbbec Femto Bolt
  - state：concat([joints, eef, hand])，18 维
  - prompt：文本任务描述

  ### 2.3 当前 action 解释方式

  当前脚本把输出动作解释为：

  - 前 6 维：joint delta
  - 后 6 维：hand delta

  也就是：

  target_joints = current_joints + action[:6]
  target_hand = current_hand + action[6:12]

  ### 2.4 启动前你需要确认

  - 目标机已经装好真机相关 Python 包和 SDK
  - 你已经有可用 PI05 checkpoint
  - 你知道机器人 IP
  - 你知道 Inspire hand 的串口路径


   ### 2.5 启动命令

  如果你走的是独立环境路线，推荐这样启动：

  source /path/to/miniconda3/etc/profile.d/conda.sh
  conda activate pi05
  source /path/to/pi05/scripts/use_local_openpi_env_standalone.sh

  python scripts/pi05_real_robot_infer.py \
    --checkpoint-dir /path/to/pi05_model_checkpoint/60000 \
    --train-config pi05_pickplace_full_pytorch \
    --prompt "place the cookie at the left top position on the desk" \
    --robot-ip 192.168.1.109 \
    --hand-port /dev/ttyUSB0

  如果你沿用当前老环境路线，则把 source scripts/use_local_openpi_env_standalone.sh 换成：

  source scripts/use_local_openpi_env.sh

  ### 2.6 首次上真机建议

  第一次不要直接高速跑，建议把参数压低：

  python scripts/pi05_real_robot_infer.py \
    --checkpoint-dir /path/to/pi05_model_checkpoint/60000 \
    --train-config pi05_pickplace_full_pytorch \
    --prompt "place the cookie at the left top position on the desk" \
    --robot-ip 192.168.1.109 \
    --hand-port /dev/ttyUSB0 \
    --control-hz 5 \
    --arm-speed 0.03 \
    --arm-acceleration 0.03

  ### 2.7 当前真机链路的边界

  现在这条 PI05 真机链路是“最小可用版”，已经能跑，但还不是完全工程化版本。

  当前还缺：

  - 关节限位保护
  - 工作空间边界保护
  - 碰撞保护
  - dry-run 模式
  - 启动前自检脚本
  - 更细的异常恢复

  所以建议：

  - 先做人工盯守下的小步验证
  - 不要直接长时间无保护运行

  ———

  ## 3. 如何跑 RECAP 的真机强化学习

  这里的“RECAP 真机强化学习”按当前仓库里的最小工作流来理解：

  1. 用已有 PI05 policy 在真机上采集 rollout
  2. 生成 RECAP 所需的数据 / returns / value 训练输入
  3. 使用你自己准备好的 value model 继续 RECAP 流程

  ### 3.1 当前 RECAP 相关目录

  你主要会用到这些位置：

  - scripts/pi05_recap_real_collect.py
  - recap_workspace/
  - pi06_recap/

  ### 3.2 真机采集入口脚本

  真机 rollout 采集脚本是：

  scripts/pi05_recap_real_collect.py

  这个脚本本质上会：

  - 加载 PI05 policy checkpoint
  - 读取真机 observation
  - 执行动作
  - 把每一帧保存下来
  - 保存每个 episode 的 frames.npz、meta.json、thumb.png

  ### 3.3 采集输出默认目录

  默认会写到：

  /nfs_global/S/yangrongzheng/pi05/data/recap_real_collect

  新机器上你可以改成自己的路径：

  --output-dir /path/to/recap_real_collect

  ### 3.4 采集命令

  source /path/to/miniconda3/etc/profile.d/conda.sh
  conda activate pi05
  source /path/to/pi05/scripts/use_local_openpi_env_standalone.sh

  python scripts/pi05_recap_real_collect.py \
    --checkpoint-dir /path/to/pi05_model_checkpoint/60000 \
    --train-config pi05_pickplace_full_pytorch \
    --prompt "place the cookie at the left top position on the desk" \
    --output-dir /path/to/recap_real_collect \
    --robot-ip 192.168.1.109 \
    --hand-port /dev/ttyUSB0 \
    --num-episodes 1 \
    --max-steps 100

  常用参数：

  - --episode-start-index：从哪个 episode 编号开始
  - --num-episodes：采多少条 episode
  - --max-steps：每条轨迹最大步数
  - --overwrite：覆盖已有输出目录

  ### 3.5 采集后的数据长什么样

  每个 episode 目录下会有：

  - frames.npz
  - meta.json
  - thumb.png

  这些是后续 RECAP 处理的原始输入。

  ### 3.6 RECAP 后续工作流

  当前仓库里的 RECAP 代码主要在：

  - recap_workspace/
  - pi06_recap/

  按当前代码结构，后续通常会分几步：

  1. 基于 rollout 生成 returns / sparse returns / labels
  2. 准备 value model 训练或打分输入
  3. 跑 value model / score / advantage 相关脚本
  4. 再把结果并回 RECAP 工作流

  ### 3.7 你需要重点看的 RECAP 文件

  #### recap_workspace/

  这个目录更偏工作流和配置：

  - recap_workspace/README.md
  - recap_workspace/docs/USAGE.md
  - recap_workspace/run_value_local.py
  - recap_workspace/patch_lerobot_runtime.py
  - recap_workspace/configs/*.yaml
  - recap_workspace/scripts/*.slurm

  #### pi06_recap/

  这个目录更偏 value / recap 逻辑实现：

  - pi06_recap/train_vf.py
  - pi06_recap/score_vf.py
  - pi06_recap/advantage.py
  - pi06_recap/labels.py
  - pi06_recap/manifest.py
  - pi06_recap/vf_data.py
  - pi06_recap/vf_model.py

  ### 3.8 当前这部分你要怎么理解

  因为你已经说了：

  - PI05 model 你会自己下载
  - RECAP 所需的 value model 你也会自己下载

  所以当前最现实的工作流是：

  1. 先让环境通
  2. 先让 PI05 真机 policy 能跑
  3. 用 scripts/pi05_recap_real_collect.py 在真机上采集 rollout
  4. 再把 rollout 喂给 recap_workspace/ 和 pi06_recap/ 里的脚本
  5. 用你自己的 value model 继续做 RECAP 强化学习流程

  ### 3.9 当前 RECAP 真机强化学习的边界

  需要明确一点：

  - 当前仓库里已经有 RECAP 相关代码和真机采集入口
  - 但“完整自动化的一键真机 RL pipeline”还没有被整理成单条命令

  也就是说，现在已经具备：

  - 真机 rollout 采集
  - value function / returns / labels / advantage 相关实现
  - 本地 / slurm 运行脚本

  但你在真正跑 RECAP 真机强化学习时，仍然需要根据自己的：

  - value model 路径
  - rollout 数据位置
  - 目标训练策略
  - 本地还是集群运行方式

  做少量参数拼接。

  ———

  ## 最后建议的实际顺序

  如果你现在要开始干活，我建议按这个顺序：

  ### 第一步：先把环境跑通

  conda create -n pi05 python=3.11 -y
  conda activate pi05
  cd openpi_official && pip install -e . && cd ..
  pip install -r requirements-standalone.txt
  source scripts/use_local_openpi_env_standalone.sh
  python scripts/check_standalone_openpi_env.py

  ### 第二步：先跑 PI05 真机推理


   确认：

  - PI05 checkpoint 可加载
  - UR5e / hand / camera 都能连上
  - 真机 observation 正常

  ### 第三步：再跑 RECAP 真机采集

  用：

  scripts/pi05_recap_real_collect.py

  先采几条短 rollout。

  ### 第四步：再接 RECAP value model

  确认 rollout 存储格式没问题后，再进入：

  - recap_workspace/
  - pi06_recap/

  继续做 returns / value / advantage / training。
