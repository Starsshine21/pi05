#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from recap_workspace.pi06_recap.labels import value_to_bin


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert adjust_bottle raw rollouts into a ReCap-style manifest jsonl.')
    parser.add_argument('--input-dir', default='/nfs_global/S/yangrongzheng/pi05/data/recap_adjust_bottle_sim/raw_rollouts')
    parser.add_argument('--output', default='/nfs_global/S/yangrongzheng/pi05/data/recap_adjust_bottle_sim/manifest.jsonl')
    parser.add_argument('--image-dir', default='/nfs_global/S/yangrongzheng/pi05/data/recap_adjust_bottle_sim/manifest_images')
    parser.add_argument('--sample-stride', type=int, default=1)
    parser.add_argument('--success-value', type=float, default=0.0)
    parser.add_argument('--fail-value', type=float, default=-1.0)
    parser.add_argument('--num-bins', type=int, default=201)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    image_dir = Path(args.image_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep_dir in sorted(p for p in input_dir.glob('episode_*') if p.is_dir()):
        meta = json.loads((ep_dir / 'meta.json').read_text())
        frames = np.load(ep_dir / 'frames.npz')
        num_frames = int(frames['state'].shape[0])
        outcome_value = args.success_value if bool(meta['success']) else args.fail_value
        value_bin = value_to_bin(outcome_value, num_bins=args.num_bins)
        for frame_index in range(0, num_frames, args.sample_stride):
            image_path = image_dir / f"episode_{int(meta['episode_index']):06d}_frame_{frame_index:06d}.png"
            if not image_path.exists():
                bgr = cv2.cvtColor(frames['image'][frame_index], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), bgr)
            rows.append({
                'episode_index': int(meta['episode_index']),
                'frame_index': int(frame_index),
                'prompt': str(meta['prompt']),
                'image_path': str(image_path),
                'success': bool(meta['success']),
                'task_name': str(meta['task_name']),
                'task_config': str(meta['task_config']),
                'train_config': str(meta['train_config']),
                'model_name': str(meta['model_name']),
                'seed': int(meta['seed']),
                'num_frames': int(meta['num_frames']),
                'timestamp': float(frames['timestamp'][frame_index]),
                'value': float(outcome_value),
                'value_bin': int(value_bin),
            })

    with output.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Wrote {len(rows)} rows to {output}')


if __name__ == '__main__':
    main()
