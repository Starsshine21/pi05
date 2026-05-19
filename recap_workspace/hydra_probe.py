import os
import runpy
import sys
import time


def mark(msg: str) -> None:
    print(f"[hydra-probe] {time.strftime('%F %T')} {msg}", flush=True)

mark('start')
os.environ.setdefault('HF_HOME', '/nfs_global/S/yangrongzheng/pi05/.cache/huggingface')
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(os.environ['HF_HOME'], 'datasets'))
train_value = '/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/rlinf-recap/examples/recap/value/train_value.py'
sys.argv = [train_value, '--help']
mark(f'argv={sys.argv}')
runpy.run_path(train_value, run_name='__main__')
mark('done')
