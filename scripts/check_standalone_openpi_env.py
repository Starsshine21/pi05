#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import pathlib
import sys

CHECKS = [
    "cv2",
    "numpy",
    "torch",
    "serial",
    "openpi.training.config",
    "openpi.policies.policy_config",
]

OPTIONAL = [
    "pyrealsense2",
    "pyorbbecsdk",
    "rtde_control",
    "rtde_receive",
]


def check_import(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"python={sys.executable}")
    print(f"cwd={os.getcwd()}")
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"OPENPI_DATA_HOME={os.environ.get('OPENPI_DATA_HOME', '')}")

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    print(f"repo_root={repo_root}")

    failed = False
    print("\n[required]")
    for name in CHECKS:
        ok, msg = check_import(name)
        print(f"- {name}: {msg}")
        failed = failed or not ok

    print("\n[optional-hardware]")
    for name in OPTIONAL:
        ok, msg = check_import(name)
        print(f"- {name}: {msg}")

    try:
        from openpi.training import config as train_config_lib
        print("\n[config]")
        print("- pi05_pickplace_full_pytorch:", "pi05_pickplace_full_pytorch" in train_config_lib._CONFIGS_DICT)
        print("- pi05_pickplace_lora:", "pi05_pickplace_lora" in train_config_lib._CONFIGS_DICT)
    except Exception as exc:
        print(f"\n[config]\n- failed: {type(exc).__name__}: {exc}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
