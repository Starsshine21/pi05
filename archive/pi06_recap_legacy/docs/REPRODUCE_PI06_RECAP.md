# π0.6-Style RECAP 复现流程

主参考：Physical Intelligence 的 π0.6 论文 PDF `https://www.physicalintelligence.company/download/pistar06.pdf`、官方 openpi 仓库 `https://github.com/Physical-Intelligence/openpi`，以及 Evo-RL 的 LeRobot/ACP 实现 `https://github.com/MINT-SJTU/Evo-RL`。

## 0. 你需要准备什么

服务器建议：

- Ubuntu 22.04 或 glibc >= 2.31。
- NVIDIA GPU。VF 训练建议 A6000 48G 级别；openpi π0.5 LoRA 微调建议 >=24G；全参微调建议 A100/H100 80G。
- 一个 LeRobot 格式数据集，目录里应有 `data/`、`videos/`、`meta/`。
- 每个 episode 的成功标注 CSV/JSONL。

成功标注 CSV 示例：

```csv
episode_index,success,success_frame
0,true,183
1,false,
2,true,221
```

`success_frame` 可省略；省略时成功 episode 默认最后一帧成功。失败 episode 的 value 固定为 -1。

## 1. 安装 VF 环境

```bash
cd /path/to/pi06_recap_kit
python3 -m venv .venv-vf
source .venv-vf/bin/activate
pip install -U pip
pip install -r requirements-vf.txt
```

如果 Hugging Face 上的 Gemma 需要授权，先登录：

```bash
huggingface-cli login
```

## 2. 生成 value function manifest

```bash
python scripts/make_value_manifest.py \
  --lerobot-root /data/lerobot/my_dataset \
  --outcomes /data/labels/outcomes.csv \
  --output /data/pi06/manifests/vf_train.jsonl \
  --image-key observation.images.top \
  --max-steps 500 \
  --sample-stride 1
```

`--image-key` 不填时会自动选 `videos/chunk-*/observation.images.*` 里的第一个相机。每条 manifest 会包含视频路径、帧号、task prompt、NTTG value 和 201-bin label。默认建议 `--sample-stride 1`，因为后面的 Evo-RL-style n-step advantage 是按 manifest 行序计算的；如果加大 stride，`--n-step` 的实际时间跨度也会随之变长。

## 3. 训练 value function

推荐先做 projector 对齐，再做机器人 VF：

```bash
python scripts/train_vf.py \
  --stage alignment \
  --output-dir /data/pi06/outputs/vf \
  --alignment-steps 5000 \
  --alignment-batch-size 8 \
  --load-in-4bit
```

然后训练 robot VF：

```bash
python scripts/train_vf.py \
  --stage robot \
  --manifest /data/pi06/manifests/vf_train.jsonl \
  --projector-path /data/pi06/outputs/vf/alignment/projector.pt \
  --output-dir /data/pi06/outputs/vf \
  --robot-epochs 5 \
  --robot-batch-size 8 \
  --use-lora \
  --load-in-4bit
```

也可以两阶段连跑：

```bash
python scripts/train_vf.py \
  --stage both \
  --manifest /data/pi06/manifests/vf_train.jsonl \
  --output-dir /data/pi06/outputs/vf \
  --alignment-steps 5000 \
  --robot-epochs 5 \
  --load-in-4bit
```

最佳 checkpoint 默认在：

```text
/data/pi06/outputs/vf/robot/best/
```

## 4. 打分并导出 advantage label

```bash
python scripts/score_vf.py \
  --manifest /data/pi06/manifests/vf_train.jsonl \
  --checkpoint /data/pi06/outputs/vf/robot/best \
  --output /data/pi06/scores/vf_scores.jsonl \
  --batch-size 16 \
  --load-in-4bit
```

默认使用 Evo-RL 风格 ACP：先用 target value 构造 dense reward，再计算 n-step advantage，最后按每个 task prompt 的 top 30% 作为 positive：

```bash
python scripts/make_advantage_labels.py \
  --scores /data/pi06/scores/vf_scores.jsonl \
  --output /data/pi06/advantage/advantage_labels.jsonl \
  --mode nstep \
  --n-step 50 \
  --positive-fraction 0.30
```

如果你只想按 VF score 排序，可改用 quantile：

```bash
python scripts/make_advantage_labels.py \
  --scores /data/pi06/scores/vf_scores.jsonl \
  --output /data/pi06/advantage/advantage_labels.jsonl \
  --mode quantile \
  --positive-fraction 0.30
```

如果你想用固定阈值，例如 score >= -0.2：

```bash
python scripts/make_advantage_labels.py \
  --scores /data/pi06/scores/vf_scores.jsonl \
  --output /data/pi06/advantage/advantage_labels.jsonl \
  --mode threshold \
  --threshold -0.2
```

## 5. 准备 openpi / π0.5 policy 微调

安装并 patch 官方 openpi：

```bash
OPENPI_DIR=/data/openpi bash scripts/setup_server_openpi.sh
```

这个脚本会 checkout 我分析时用的官方 openpi commit：

```text
c23745b5ad24e98f66967ea795a07b2588ed6c79
```

patch 做的事情很小：修改 `src/openpi/transforms.py` 的 `PromptFromLeRobotTask`，如果环境变量 `PI06_ADVANTAGE_JSONL` 指向上一节生成的 label 文件，就把当前 `(episode_index, frame_index)` 对应的：

```text
Advantage: positive
```

或：

```text
Advantage: negative
```

追加到 prompt 末尾。

## 6. 配置你的 openpi 训练项

如果你用的是 openpi 已支持的数据集，比如 LIBERO/DROID/ALOHA，可以直接复用对应 config。

如果是自己的 LeRobot 数据集，需要在 `/data/openpi/src/openpi/training/config.py` 增加一个 config，核心点是：

- `repo_id` 指向你的 LeRobot dataset。
- `prompt_from_task=True`，这样 patch 才能在 task prompt 后注入 advantage。
- `action_dim` 和 `action_horizon` 必须匹配你的机器人。
- LoRA 用 `paligemma_variant="gemma_2b_lora"` 和 `action_expert_variant="gemma_300m_lora"`，并设置对应 `freeze_filter`。

hzm8341 的 G1 示例可以作为 43-DOF 单相机数据集模板：`src/openpi/policies/g1_policy.py` 和 `pi05_g1_pick_apple_lora` config。

## 7. 训练 advantage-conditioned policy

```bash
OPENPI_DIR=/data/openpi \
HF_LEROBOT_HOME=/data/lerobot \
CONFIG=pi05_your_dataset_lora \
EXP_NAME=pi06_adv_run1 \
ADVANTAGE_JSONL=/data/pi06/advantage/advantage_labels.jsonl \
bash scripts/train_openpi_advantage.sh
```

训练脚本会先计算 norm stats，再用 openpi 训练。checkpoint 会在：

```text
/data/openpi/checkpoints/<CONFIG>/<EXP_NAME>/
```

## 8. 推理

π0.6-style advantage conditioning 的推理做法是固定使用 positive condition：

```bash
OPENPI_DIR=/data/openpi \
CONFIG=pi05_your_dataset_lora \
CHECKPOINT=/data/openpi/checkpoints/pi05_your_dataset_lora/pi06_adv_run1/20000 \
bash scripts/serve_openpi_positive.sh
```

这个脚本设置：

```bash
PI06_FORCE_ADVANTAGE=positive
```

因此所有来自 LeRobot task 的 prompt 都会追加 `Advantage: positive`。

## 9. 重要限制

- 这不是官方 π0.6 代码复现。官方 π0.6 的数据、模型权重、训练细节没有开源。
- 这里复现的是论文里最容易落地的 RECAP/advantage-conditioning 主干：VF 打分 + positive conditioning policy 微调。
- 如果你的数据没有可靠成功标注，VF 会学不到有效排序。
- 当前 VF 只用一个相机训练；多相机可以扩展 manifest 和模型，把多个 SigLIP token 串接后再进 Gemma。
- 如果你要复现 MEM/长期记忆分支，hzm fork 只能作为草稿参考，还需要把多帧数据 wrapper 接入 openpi 的 `create_torch_dataset` 和训练 config。

## 10. Evo-RL 原生路线可选项

Evo-RL 已经在 LeRobot 内部实现了一套更完整的 CLI：

```bash
lerobot-value-train --value.type=pistar06 ...
lerobot-value-infer --acp.enable=true --acp.n_step=50 --acp.positive_ratio=0.3 ...
lerobot-train --acp.enable=true --acp.indicator_field=complementary_info.acp_indicator_<TAG> ...
```

如果你的策略模型也在 Evo-RL/LeRobot 体系里，直接用这条原生路线更省事。当前复现包保留 openpi 作为 policy 底座，所以只吸收 Evo-RL 的 ACP 计算方式，并通过 `PI06_ADVANTAGE_JSONL` patch 把标签喂给 openpi。
