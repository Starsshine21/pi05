#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import time

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a HuggingFace dataset file-by-file with retries.")
    parser.add_argument("repo_id")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--include-prefix", default="")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = pathlib.Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = list(api.list_repo_files(args.repo_id, repo_type=args.repo_type, revision=args.revision))
    if args.include_prefix:
        files = [path for path in files if path.startswith(args.include_prefix)]

    print(f"repo_id={args.repo_id}")
    print(f"repo_type={args.repo_type}")
    print(f"local_dir={local_dir}")
    print(f"files={len(files)}")

    failed: list[str] = []
    for index, filename in enumerate(files, start=1):
        target = local_dir / filename
        if target.exists() and target.stat().st_size > 0 and not args.force:
            print(f"[{index}/{len(files)}] skip {filename} ({target.stat().st_size} bytes)", flush=True)
            continue

        for attempt in range(1, args.retries + 1):
            try:
                print(f"[{index}/{len(files)}] download {filename} attempt={attempt}", flush=True)
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=filename,
                    repo_type=args.repo_type,
                    revision=args.revision,
                    local_dir=str(local_dir),
                    force_download=args.force,
                )
                break
            except Exception as exc:
                print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
                if attempt == args.retries:
                    failed.append(filename)
                    break
                time.sleep(args.sleep)

    if failed:
        print("\nfailed files:")
        for filename in failed:
            print(filename)
        return 1

    print("download complete")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    raise SystemExit(main())
