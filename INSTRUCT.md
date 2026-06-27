---
language:
- en
library_name: lerobot
pipeline_tag: robotics
tags:
- vision-language-action
- imitation-learning
- lerobot
inference: false
license: gemma
---

# π₀.₅ (Pi05) (LeRobot)

π₀.₅ is a Vision-Language-Action (VLA) model with open-world generalization from Physical Intelligence, co-trained on robot demonstrations and large-scale multimodal data to execute long-horizon tasks in unseen real-world environments.

**Note:** This model currently supports only the flow-matching action head for π₀.₅ training and inference. 
Other components from the original work (e.g., subtask prediction, action tokenization, or RL) were not released upstream and are not included here, though the LeRobot team is actively working to support them.

**Original paper:** π0.5: A Vision-Language-Action Model with Open-World Generalization  
**Reference implementation:** https://github.com/Physical-Intelligence/openpi  
**LeRobot implementation:** Follows the original reference code for compatibility.


## Model description

- **Inputs:** images (multi-view), proprio/state, optional language instruction
- **Outputs:** continuous actions
- **Training objective:** flow matching
- **Action representation:** continuous
- **Intended use:** Base model to fine tune on your specific use case


## Quick start (inference on a real batch)

### Installation

```bash
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
```
For full installation details (including optional video dependencies such as ffmpeg for torchcodec), see the official documentation: https://huggingface.co/docs/lerobot/installation

### Load model + dataset, run `select_action`

```python
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

# Swap this import per-policy
from lerobot.policies.pi05 import PI05Policy

# load a policy
model_id = "lerobot/pi05_libero_base"  # <- swap checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = PI05Policy.from_pretrained(model_id).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
# load a lerobotdataset (we will replace with a simpler dataset)
dataset = LeRobotDataset("lerobot/libero")

# pick an episode
episode_index = 0

# each episode corresponds to a contiguous range of frame indices
from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
to_idx   = dataset.meta.episodes["dataset_to_index"][episode_index]

# get a single frame from that episode (e.g. the first frame)
frame_index = from_idx
frame = dict(dataset[frame_index])

batch = preprocess(frame)
with torch.inference_mode():
    pred_action = policy.select_action(frame)
    # use your policy postprocess, this post process the action
    # for instance unnormalize the actions, detokenize it etc..
    pred_action = postprocess(pred_action)
```


## Training step (loss + backward)

If you’re training / fine-tuning, you typically call `forward(...)` to get a loss and then:

```python
policy.train()
batch = dict(dataset[0])
batch = preprocess(batch)

loss, outputs = policy.forward(batch)
loss.backward()

```

> Notes:
> 
> - Some policies expose `policy(**batch)` or return a dict; keep this snippet aligned with the policy API.
> - Use your trainer script (`lerobot-train`) for full training loops.


## How to train / fine-tune

```bash
lerobot-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --output_dir=./outputs/[RUN_NAME] \
  --job_name=[RUN_NAME] \
  --policy.repo_id=[THIS_REPO_OR_CHECKPOINT] \
  --policy.path=lerobot/[BASE_CHECKPOINT] \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --steps=100000 \
  --batch_size=4
```

Add policy-specific flags below:

- `-policy.chunk_size=...`
- `-policy.n_action_steps=...`
- `-policy.max_action_tokens=...`
- `-policy.gradient_checkpointing=true`


## Evaluate in Simulation (LIBERO)

You can evaluate the model in Libero environment.

```bash
lerobot-eval \
  --policy.path=lerobot/pi05_libero_base \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=1 \
  --eval.n_episodes=20
```
