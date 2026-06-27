#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Deprecated wrapper. Use RLinf compute_returns.py instead.'
    )
    parser.add_argument('--dataset-dir', help='Ignored. Kept only for compatibility.')
    parser.add_argument('--tag', default='local_pi05_online', help='Ignored. Kept only for compatibility.')
    parser.parse_args()

    root = Path('/nfs_global/S/yangrongzheng/pi05')
    python_bin = root / 'recap_workspace' / 'external' / 'venv' / 'bin' / 'python'
    cmd = [
        str(python_bin),
        str(root / 'recap_workspace' / 'external' / 'rlinf-recap' / 'examples' / 'recap' / 'process' / 'compute_returns.py'),
        '--config-path',
        str(root / 'recap_workspace' / 'configs'),
        '--config-name',
        'local_compute_returns',
    ]
    print('`scripts/generate_sparse_returns.py` is deprecated.')
    print('Using RLinf official compute_returns instead:')
    print(' '.join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=root / 'recap_workspace' / 'external' / 'rlinf-recap' / 'examples' / 'recap' / 'process'))


if __name__ == '__main__':
    main()
