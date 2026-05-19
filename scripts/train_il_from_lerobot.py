#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    print(f"Missing dependency: {exc.name}", file=sys.stderr)
    print("Please run in the pi05 environment:", file=sys.stderr)
    print("  source /home/S/yangrongzheng/miniconda3/etc/profile.d/conda.sh", file=sys.stderr)
    print("  conda activate /nfs_global/S/yangrongzheng/pi05/.conda-pi05-openpi-final", file=sys.stderr)
    print("  python scripts/train_il_from_lerobot.py --epochs 5 --batch-size 64", file=sys.stderr)
    raise

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class LocalLeRobotDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        tasks = {}
        with (self.root / "meta" / "tasks.jsonl").open() as file:
            for line in file:
                row = json.loads(line)
                tasks[row["task_index"]] = row["task"]
        self.tasks = tasks
        self.rows = []
        for parquet_path in sorted((self.root / "data").glob("chunk-*/*.parquet")):
            table = pq.read_table(parquet_path)
            self.rows.extend(table.to_pylist())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
            "action": torch.tensor(row["action"], dtype=torch.float32),
            "task": self.tasks[int(row["task_index"])],
        }


class SimpleLeRobotILPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state):
        return self.net(state)


def collate_fn(batch):
    return {
        "state": torch.stack([item["observation.state"] for item in batch]),
        "action": torch.stack([item["action"] for item in batch]),
        "task": [item["task"] for item in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="Train IL baseline from local LeRobot dataset directory.")
    parser.add_argument("--dataset-root", default="/nfs_global/S/yangrongzheng/pi05/data/lerobot_pick_place")
    parser.add_argument("--output-dir", default="/nfs_global/S/yangrongzheng/pi05/results/pi05_il_lerobot")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    dataset = LocalLeRobotDataset(args.dataset_root)
    train_size = max(1, int(len(dataset) * args.train_ratio))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) if val_size > 0 else (dataset, None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None

    sample = dataset[0]
    state_dim = sample["observation.state"].numel()
    action_dim = sample["action"].numel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLeRobotILPolicy(state_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    print(f"dataset size: {len(dataset)}")
    print(f"train/val: {len(train_dataset)}/{len(val_dataset) if val_dataset else 0}")
    print(f"state_dim/action_dim: {state_dim}/{action_dim}")
    print(f"device: {device}")

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"lerobot train epoch {epoch}")
        for batch in progress:
            state = batch["state"].to(device=device, dtype=torch.float32)
            action = batch["action"].to(device=device, dtype=torch.float32)
            pred = model(state)
            loss = loss_fn(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=float(loss.item()))

        if val_loader:
            model.eval()
            vals = []
            with torch.no_grad():
                for batch in val_loader:
                    state = batch["state"].to(device=device, dtype=torch.float32)
                    action = batch["action"].to(device=device, dtype=torch.float32)
                    pred = model(state)
                    vals.append(loss_fn(pred, action).item())
            val_loss = float(np.mean(vals)) if vals else 0.0
            print(f"epoch {epoch} val_loss={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "state_dim": state_dim, "action_dim": action_dim}, out_dir / "best.pt")

    torch.save({"model": model.state_dict(), "state_dim": state_dim, "action_dim": action_dim}, out_dir / "last.pt")
    print(f"saved checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
