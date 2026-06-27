#!/usr/bin/env python3
"""Convert LeRobot v2.1 legacy jsonl episodes to parquet format with episode_success field."""

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_episodes(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    meta_dir = root / "meta"

    episodes_jsonl = meta_dir / "episodes.jsonl"
    if not episodes_jsonl.exists():
        raise FileNotFoundError(f"episodes.jsonl not found: {episodes_jsonl}")

    print(f"Reading {episodes_jsonl}")
    rows = []
    with open(episodes_jsonl) as f:
        for line in f:
            row = json.loads(line.strip())
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} episodes")
    print(f"Original columns: {df.columns.tolist()}")

    if "episode_success" not in df.columns:
        df["episode_success"] = args.label
        print(f"Added episode_success='{args.label}' to all episodes")
    else:
        mask = df["episode_success"].isna() | (df["episode_success"].astype(str).str.strip() == "")
        if mask.any():
            df.loc[mask, "episode_success"] = args.label
            print(f"Filled {mask.sum()} missing episode_success with '{args.label}'")
        else:
            print("All episodes already have episode_success")

    print(f"Final columns: {df.columns.tolist()}")

    episodes_dir = meta_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    if args.output_root:
        target_root = args.output_root.resolve()
        if target_root.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output root exists: {target_root}. Use --overwrite to replace.")
            import shutil
            shutil.rmtree(target_root)
        import shutil
        shutil.copytree(root, target_root)
        target_episodes_dir = target_root / "meta" / "episodes"
        target_episodes_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_episodes_dir / "chunk-000" / "file-000.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")
        summary_path = target_root / "conversion_summary.json"
        summary = {
            "source_root": str(root),
            "target_root": str(target_root),
            "label": args.label,
            "total_episodes": len(df),
            "columns": df.columns.tolist(),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Summary: {summary_path}")
    else:
        output_path = episodes_dir / "chunk-000" / "file-000.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--output-root", type=Path, help="Output root (if different from input)")
    parser.add_argument("--label", choices=("success", "failure"), default="success")
    parser.add_argument("--overwrite", action="store_true")
    convert_episodes(parser.parse_args())


if __name__ == "__main__":
    main()
