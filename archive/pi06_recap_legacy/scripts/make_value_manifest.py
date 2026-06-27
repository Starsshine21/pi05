#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi06_recap.labels import read_outcomes
from pi06_recap.manifest import ManifestConfig, build_manifest_rows, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VF training manifest from a local LeRobot dataset.")
    parser.add_argument("--lerobot-root", required=True, help="Local LeRobot dataset root containing data/, videos/, meta/.")
    parser.add_argument("--outcomes", required=True, help="CSV/JSONL with episode_index, success, optional success_frame.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest.")
    parser.add_argument("--image-key", default=None, help="Camera key, e.g. observation.images.top. Auto-detected if omitted.")
    parser.add_argument("--max-steps", type=int, default=500, help="NTTG normalization horizon.")
    parser.add_argument("--num-bins", type=int, default=201)
    parser.add_argument("--sample-stride", type=int, default=1, help="Keep every Nth frame for VF training/scoring.")
    parser.add_argument("--default-prompt", default="", help="Fallback task prompt if metadata has no task text.")
    parser.add_argument("--failed-value", type=float, default=-1.0)
    args = parser.parse_args()

    outcomes = read_outcomes(args.outcomes)
    rows = build_manifest_rows(
        ManifestConfig(
            lerobot_root=Path(args.lerobot_root),
            image_key=args.image_key,
            max_steps=args.max_steps,
            num_bins=args.num_bins,
            sample_stride=args.sample_stride,
            default_prompt=args.default_prompt,
            failed_value=args.failed_value,
        ),
        outcomes,
    )
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
