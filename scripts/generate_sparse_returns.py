#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate sparse terminal returns sidecar for a LeRobot dataset.')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--tag', default='local_pi05_online')
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    episodes = root / 'meta' / 'episodes.jsonl'
    out = root / 'meta' / f'returns_{args.tag}.parquet'
    rows = []
    with episodes.open() as f:
        for line in f:
            rec = json.loads(line)
            ep = int(rec['episode_index'])
            length = int(rec['length'])
            task = rec.get('tasks', [''])[0]
            success = 'fail' not in task.lower()
            for frame in range(length):
                reward = 1.0 if success and frame == length - 1 else 0.0
                ret = 1.0 if success else 0.0
                rows.append({'episode_index': ep, 'frame_index': frame, 'return': ret, 'reward': reward})
    df = pd.DataFrame(rows, columns=['episode_index', 'frame_index', 'return', 'reward'])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(out)


if __name__ == '__main__':
    main()
