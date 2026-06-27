#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoProcessor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from rlinf.models import get_model
from omegaconf import OmegaConf


class Pi05ILWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(**batch)


def collate_fn(samples):
    batch = {}
    for key in samples[0]:
        vals = [sample[key] for sample in samples]
        if torch.is_tensor(vals[0]):
            batch[key] = torch.stack(vals)
        else:
            batch[key] = vals
    return batch


def make_model(model_dir: str):
    cfg = OmegaConf.create(
        {
            "model_name": "openpi",
            "precision": None,
            "num_action_chunks": 10,
            "action_dim": 12,
            "add_value_head": False,
            "is_lora": False,
            "lora_rank": 32,
            "gradient_checkpointing": False,
            "use_wrist_image": True,
            "use_proprio": True,
            "openpi": {
                "simulator_type": "libero",
                "pi05": True,
                "noise_level": 0.5,
                "action_chunk": 10,
                "num_steps": 5,
                "train_expert_only": True,
                "action_env_dim": 12,
                "noise_method": "flow_sde",
                "add_value_head": False,
            },
        }
    )
    return get_model(model_dir, cfg)


def main():
    parser = argparse.ArgumentParser(description="Minimal IL finetuning on local LeRobot pick-place dataset.")
    parser.add_argument("--dataset-root", default="/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place")
    parser.add_argument("--repo-id", default="local/pi05-pickplace-il")
    parser.add_argument("--model-dir", default="/nfs_global/S/yangrongzheng/pi05/pi05_sft")
    parser.add_argument("--output-dir", default="/nfs_global/S/yangrongzheng/pi05/results/pi05_il")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)
    if args.max_samples < len(dataset):
        dataset, _ = random_split(dataset, [args.max_samples, len(dataset) - args.max_samples])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = make_model(args.model_dir).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            global_step += 1
            obs_state = batch["observation.state"].to(device=device, dtype=torch.float32)
            target_action = batch["action"].to(device=device, dtype=torch.float32)

            if target_action.shape[-1] != 12:
                target_action = target_action[..., :12]
            if obs_state.ndim == 1:
                obs_state = obs_state.unsqueeze(0)

            pred = obs_state[:, : target_action.shape[-1]]
            loss = mse(pred, target_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()), step=global_step)

    ckpt = output_dir / "il_stub.pt"
    torch.save({"model_dir": args.model_dir, "note": "stub IL trainer baseline finished"}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
