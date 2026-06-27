#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path

mods = [
    'sapien',
    'mplib',
    'gymnasium',
    'open3d',
    'trimesh',
]

for name in mods:
    try:
        importlib.import_module(name)
        print(f'{name}: ok')
    except Exception as exc:
        print(f'{name}: FAIL: {type(exc).__name__}: {exc}')
        sys.exit(1)

root = Path('/nfs_global/S/yangrongzheng/pi05/external/RoboTwin')
assets = Path('/nfs_global/S/yangrongzheng/pi05/../assets')
print(f'robotwin_root_exists={root.exists()}')
print(f'assets_embodiments_exists={(assets / "embodiments").exists()}')
print(f'assets_objects_exists={(assets / "objects").exists()}')
