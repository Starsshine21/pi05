#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'openpi_official' / 'src'))

from openpi.policies import policy_config as policy_config_lib
from openpi.training import config as train_config_lib


def decode_image(cell) -> np.ndarray:
    if isinstance(cell, dict):
        if 'bytes' in cell and cell['bytes'] is not None:
            return np.asarray(Image.open(io.BytesIO(cell['bytes'])).convert('RGB'))
        if 'path' in cell and cell['path']:
            return np.asarray(Image.open(cell['path']).convert('RGB'))
    if isinstance(cell, bytes):
        return np.asarray(Image.open(io.BytesIO(cell)).convert('RGB'))
    raise TypeError(f'Unsupported image cell type: {type(cell)}')


def load_row(data_dir: pathlib.Path, episode_index: int, frame_index: int):
    parquet = data_dir / 'data' / 'chunk-000' / f'episode_{episode_index:06d}.parquet'
    df = pd.read_parquet(parquet)
    row = df.iloc[frame_index]
    prompt = row['prompt'][0] if isinstance(row['prompt'], (list, tuple, np.ndarray)) else row['prompt']
    obs = {
        'observation/image': decode_image(row['image']),
        'observation/wrist_image': decode_image(row['wrist_image']),
        'observation/state': np.asarray(row['state'], dtype=np.float32),
        'prompt': prompt,
    }
    gt = np.asarray(row['actions'], dtype=np.float32)
    return parquet, obs, gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=pathlib.Path, default=REPO_ROOT / 'data' / 'lerobot_pick_place')
    parser.add_argument('--checkpoint-dir', type=pathlib.Path, required=True)
    parser.add_argument('--train-config', default='pi05_pickplace_full_pytorch')
    parser.add_argument('--episode-index', type=int, default=0)
    parser.add_argument('--frame-index', type=int, default=0)
    parser.add_argument('--output-json', type=pathlib.Path, default=None)
    args = parser.parse_args()

    parquet, obs, gt = load_row(args.data_dir, args.episode_index, args.frame_index)
    cfg = train_config_lib._CONFIGS_DICT[args.train_config]

    t0 = time.time()
    policy = policy_config_lib.create_trained_policy(cfg, args.checkpoint_dir, default_prompt=obs['prompt'])
    load_s = time.time() - t0

    t1 = time.time()
    out = policy.infer(obs)
    infer_s = time.time() - t1

    pred = np.asarray(out['actions'])
    pred_first = pred.reshape(-1, pred.shape[-1])[0]
    result = {
        'parquet': str(parquet),
        'episode_index': args.episode_index,
        'frame_index': args.frame_index,
        'prompt': obs['prompt'],
        'image_shape': list(obs['observation/image'].shape),
        'wrist_image_shape': list(obs['observation/wrist_image'].shape),
        'state_shape': list(obs['observation/state'].shape),
        'policy_load_sec': load_s,
        'policy_infer_sec': infer_s,
        'pred_action_shape': list(pred.shape),
        'pred_action_first12': pred_first[:12].tolist(),
        'gt_action_first12': gt[:12].tolist(),
        'policy_timing': out.get('policy_timing', {}),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
