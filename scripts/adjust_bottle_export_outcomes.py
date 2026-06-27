#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Export episode outcomes from adjust_bottle raw rollouts.')
    parser.add_argument('--input-dir', default='/nfs_global/S/yangrongzheng/pi05/data/recap_adjust_bottle_sim/raw_rollouts')
    parser.add_argument('--output', default='/nfs_global/S/yangrongzheng/pi05/data/recap_adjust_bottle_sim/outcomes.jsonl')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for ep_dir in sorted(p for p in input_dir.glob('episode_*') if p.is_dir()):
        meta = json.loads((ep_dir / 'meta.json').read_text())
        rows.append({
            'episode_index': int(meta['episode_index']),
            'success': bool(meta['success']),
            'success_frame': int(meta['num_frames'] - 1) if bool(meta['success']) else None,
            'prompt': str(meta['prompt']),
            'seed': int(meta['seed']),
        })

    with output.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Wrote {len(rows)} rows to {output}')


if __name__ == '__main__':
    main()
