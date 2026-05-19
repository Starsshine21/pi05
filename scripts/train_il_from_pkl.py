#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class PickPlacePKLDataset(Dataset):
    def __init__(self, root: str, action_mode: str = 'joint_and_gripper_delta', max_episodes: int | None = None):
        self.files = sorted(Path(root).glob('*.pkl'))
        if max_episodes is not None:
            self.files = self.files[:max_episodes]
        self.action_mode = action_mode
        self.samples = []
        self._build_index()

    def _compute_action(self, joints, eef, hand):
        if self.action_mode == 'joint_delta':
            base = joints
        elif self.action_mode == 'eef_delta':
            base = np.concatenate([eef, hand], axis=1)
        else:
            base = np.concatenate([joints, hand], axis=1)
        action = np.zeros_like(base, dtype=np.float32)
        action[:-1] = base[1:] - base[:-1]
        action[-1] = action[-2] if len(action) > 1 else 0
        return action

    def _build_index(self):
        for file_path in self.files:
            with file_path.open('rb') as f:
                data = pickle.load(f)
            joints = np.asarray(data['episode_ur5e_pos_j'], dtype=np.float32)
            eef = np.asarray(data['episode_ur5e_pos_eef'], dtype=np.float32)
            hand = np.asarray(data['episode_inspire_hand_pos'], dtype=np.float32)
            cam0 = np.asarray(data['episode_l515_color'], dtype=np.uint8)
            cam1 = np.asarray(data['episode_orbbec_femto_bolt_color'], dtype=np.uint8)
            state = np.concatenate([joints, eef, hand], axis=1).astype(np.float32)
            action = self._compute_action(joints, eef, hand).astype(np.float32)
            stem = file_path.stem
            task = stem.split('_', 1)[1] if '_' in stem else stem
            for t in range(len(state)):
                self.samples.append({
                    'state': state[t],
                    'action': action[t],
                    'image0': cam0[t],
                    'image1': cam1[t],
                    'task': task,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'state': torch.from_numpy(item['state']),
            'action': torch.from_numpy(item['action']),
            'image0': torch.from_numpy(item['image0']).permute(2,0,1).float() / 255.0,
            'image1': torch.from_numpy(item['image1']).permute(2,0,1).float() / 255.0,
            'task': item['task'],
        }


class SimpleILPolicy(nn.Module):
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
        'state': torch.stack([x['state'] for x in batch]),
        'action': torch.stack([x['action'] for x in batch]),
        'image0': torch.stack([x['image0'] for x in batch]),
        'image1': torch.stack([x['image1'] for x in batch]),
        'task': [x['task'] for x in batch],
    }


def main():
    parser = argparse.ArgumentParser(description='Train a direct IL baseline from raw PKL episodes.')
    parser.add_argument('--input-dir', default='/nfs_global/S/yangrongzheng/pick_place_raw_data')
    parser.add_argument('--output-dir', default='/nfs_global/S/yangrongzheng/pi05/results/pi05_il_raw')
    parser.add_argument('--action-mode', choices=['joint_delta','eef_delta','joint_and_gripper_delta'], default='joint_and_gripper_delta')
    parser.add_argument('--max-episodes', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-ratio', type=float, default=0.95)
    args = parser.parse_args()

    ds = PickPlacePKLDataset(args.input_dir, action_mode=args.action_mode, max_episodes=args.max_episodes)
    train_size = max(1, int(len(ds) * args.train_ratio))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size]) if val_size > 0 else (ds, None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) if val_ds else None

    sample = ds[0]
    state_dim = sample['state'].numel()
    action_dim = sample['action'].numel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleILPolicy(state_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'train epoch {epoch}')
        for batch in pbar:
            state = batch['state'].to(device=device, dtype=torch.float32)
            action = batch['action'].to(device=device, dtype=torch.float32)
            pred = model(state)
            loss = loss_fn(pred, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        if val_loader:
            model.eval()
            vals = []
            with torch.no_grad():
                for batch in val_loader:
                    state = batch['state'].to(device=device, dtype=torch.float32)
                    action = batch['action'].to(device=device, dtype=torch.float32)
                    pred = model(state)
                    vals.append(loss_fn(pred, action).item())
            val_loss = float(np.mean(vals)) if vals else 0.0
            print(f'epoch {epoch} val_loss={val_loss:.6f}')
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'model': model.state_dict(), 'state_dim': state_dim, 'action_dim': action_dim}, out / 'best.pt')

    torch.save({'model': model.state_dict(), 'state_dim': state_dim, 'action_dim': action_dim}, out / 'last.pt')
    print(f'saved checkpoints to {out}')


if __name__ == '__main__':
    main()
