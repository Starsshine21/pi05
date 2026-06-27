#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys


REQUIRED_IMPORTS = [
    "numpy",
    "torch",
    "jax",
    "openpi.training.config",
    "openpi.policies.policy_config",
    "openpi_client",
    "robosuite",
    "bddl",
    "mujoco",
    "libero",
    "libero.libero.benchmark",
    "libero.libero.envs",
]


def check_import(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception as exc:
        print(f"- {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"- {name}: ok")
    return True


def check_libero_paths() -> bool:
    from libero.libero import get_libero_path

    ok = True
    print("\n[libero-paths]")
    for key in ["benchmark_root", "bddl_files", "init_states", "assets", "datasets"]:
        try:
            path = pathlib.Path(get_libero_path(key))
        except Exception as exc:
            print(f"- {key}: {type(exc).__name__}: {exc}")
            ok = False
            continue
        exists = path.exists()
        print(f"- {key}: {path} ({'ok' if exists else 'missing'})")
        ok = ok and exists
    return ok


def check_openpi_config() -> bool:
    from openpi.training import config as train_config_lib

    print("\n[openpi-config]")
    try:
        cfg = train_config_lib.get_config("pi05_libero")
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    except Exception as exc:
        print(f"- pi05_libero: {type(exc).__name__}: {exc}")
        return False

    print("- pi05_libero: ok")
    print(f"- repo_id: {data_cfg.repo_id}")
    print(f"- assets_dirs: {cfg.assets_dirs}")
    print(f"- checkpoint_base_dir: {cfg.checkpoint_base_dir}")
    return data_cfg.repo_id == "physical-intelligence/libero"


def check_env_smoke() -> bool:
    from libero.libero import benchmark
    from libero.libero import get_libero_path

    print("\n[libero-smoke]")
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict["libero_10"]()
    task = suite.get_task(0)
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = suite.get_task_init_states(0)
    print(f"- suite: libero_10, tasks={suite.n_tasks}")
    print(f"- first_task: {task.language}")
    print(f"- bddl: {bddl_path} ({'ok' if bddl_path.exists() else 'missing'})")
    print(f"- init_states: shape={getattr(init_states, 'shape', None)}")
    return bddl_path.exists() and len(init_states) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local pi05 LIBERO simulation environment.")
    parser.add_argument("--env-smoke", action="store_true", help="Load a LIBERO benchmark task and initial states.")
    args = parser.parse_args()

    print(f"python={sys.executable}")
    print(f"cwd={os.getcwd()}")
    for key in [
        "PYTHONPATH",
        "LIBERO_CONFIG_PATH",
        "LIBERO_DATASETS_DIR",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "HF_HOME",
        "HF_LEROBOT_HOME",
        "OPENPI_DATA_HOME",
        "NUMBA_CACHE_DIR",
    ]:
        print(f"{key}={os.environ.get(key, '')}")

    failed = False
    print("\n[imports]")
    for name in REQUIRED_IMPORTS:
        failed = (not check_import(name)) or failed

    if not failed:
        failed = (not check_libero_paths()) or failed
        failed = (not check_openpi_config()) or failed
        if args.env_smoke:
            failed = (not check_env_smoke()) or failed

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
