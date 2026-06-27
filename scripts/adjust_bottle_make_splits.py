#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recap_workspace.pi06_recap.vf_data import load_jsonl, split_rows_by_episode


def dump_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Split adjust_bottle manifest rows by episode.')
    parser.add_argument('--manifest', default='/nfs_global/S/yangrongzheng/pi05/data/recap_adjust_bottle_sim/manifest.jsonl')
    parser.add_argument('--output-dir', default='/nfs_global/S/yangrongzheng/pi05/data/recap_adjust_bottle_sim/splits')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    train_rows, val_rows, test_rows = split_rows_by_episode(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    out = Path(args.output_dir)
    dump_rows(out / 'train.jsonl', train_rows)
    dump_rows(out / 'val.jsonl', val_rows)
    dump_rows(out / 'test.jsonl', test_rows)
    summary = {
        'train_rows': len(train_rows),
        'val_rows': len(val_rows),
        'test_rows': len(test_rows),
        'train_episodes': len({int(r['episode_index']) for r in train_rows}),
        'val_episodes': len({int(r['episode_index']) for r in val_rows}),
        'test_episodes': len({int(r['episode_index']) for r in test_rows}),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
