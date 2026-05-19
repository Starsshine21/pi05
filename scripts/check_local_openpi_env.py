#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
import sys
import traceback


CHECKS = [
    ("numpy", "numpy"),
    ("jax", "jax"),
    ("flax", "flax"),
    ("lerobot", "lerobot"),
    ("openpi", "openpi"),
    ("torch", "torch"),
    ("openpi.training.config", "openpi.training.config"),
    ("rlinf", "rlinf"),
    ("libero", "libero"),
    ("rlinf.models.embodiment.openpi", "rlinf.models.embodiment.openpi"),
]


def main() -> int:
    print(f"python={sys.executable}")
    print(f"python_version={sys.version.split()[0]}")
    print(f"cuda_runtime_root={os.environ.get('CUDA_RUNTIME_ROOT', '')}")

    failures = 0
    for label, module_name in CHECKS:
        try:
            module = importlib.import_module(module_name)
            location = getattr(module, "__file__", "<built-in>")
            print(f"[OK]   {label}: {location}")
            if module_name == "torch":
                print(f"       torch_version={module.__version__}")
                print(f"       cuda_available={module.cuda.is_available()}")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {label}: {exc.__class__.__name__}: {exc}")
            print(traceback.format_exc().rstrip())

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
