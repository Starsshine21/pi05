#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import sys


def configure_pythonpath(repo_root: pathlib.Path) -> None:
    robotwin_root = repo_root / "external" / "RoboTwin"
    extra_paths = [
        robotwin_root,
        robotwin_root / "policy" / "pi05" / "src",
        robotwin_root / "policy" / "pi05" / "packages" / "openpi-client" / "src",
    ]
    for path in reversed(extra_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    for module_name in list(sys.modules):
        if module_name == "openpi" or module_name.startswith("openpi."):
            del sys.modules[module_name]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check RobotWin PI05 local package/config/norm-stats wiring.")
    parser.add_argument(
        "--repo-root",
        default="/nfs_global/S/yangrongzheng/pi05",
        help="Workspace root containing external/RoboTwin.",
    )
    parser.add_argument("--train-config", default="pi05_aloha_full_base")
    parser.add_argument("--model-name", default="model_robotwin")
    parser.add_argument("--checkpoint-id", default="30000")
    parser.add_argument("--asset-id", default="robotwin")
    args = parser.parse_args()

    repo_root = pathlib.Path(args.repo_root).resolve()
    robotwin_root = repo_root / "external" / "RoboTwin"
    checkpoint_dir = robotwin_root / "policy" / "pi05" / "checkpoints" / args.train_config / args.model_name / args.checkpoint_id

    print(f"python={sys.executable}")
    print(f"repo_root={repo_root}")
    print(f"robotwin_root={robotwin_root}")
    print(f"checkpoint_dir={checkpoint_dir}")

    configure_pythonpath(repo_root)
    os.chdir(robotwin_root)

    import openpi
    from openpi.training import checkpoints as checkpoint_lib
    from openpi.training import config as train_config_lib

    print(f"openpi={openpi.__file__}")

    norm_stats = checkpoint_lib.load_norm_stats(checkpoint_dir / "assets", args.asset_id)
    print(f"norm_stats_keys={sorted(norm_stats.keys())}")
    print(f"state_len={len(norm_stats['state'].mean)}")
    print(f"actions_len={len(norm_stats['actions'].mean)}")

    cfg = train_config_lib.get_config(args.train_config)
    print(f"config={cfg.name}")
    print(f"model_type={type(cfg.model).__name__}")

    expected_openpi_root = robotwin_root / "policy" / "pi05" / "src" / "openpi"
    if pathlib.Path(openpi.__file__).resolve().parent != expected_openpi_root:
        raise RuntimeError(f"Expected local openpi at {expected_openpi_root}, got {openpi.__file__}")

    if len(norm_stats["state"].mean) != 14 or len(norm_stats["actions"].mean) != 14:
        raise RuntimeError("Expected robotwin norm stats to be 14-D state/action for current RoboTwin qpos setup.")

    print("robotwin-pi05 smoke check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
