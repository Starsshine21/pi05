import os
import sys
import time


def mark(msg: str) -> None:
    print(f"[probe] {time.strftime('%F %T')} {msg}", flush=True)


mark('python start')
mark(f'python={sys.executable}')
mark(f'cwd={os.getcwd()}')
mark(f'HF_HOME={os.environ.get("HF_HOME")}')
mark(f'HF_DATASETS_CACHE={os.environ.get("HF_DATASETS_CACHE")}')

mark('import torch: start')
import torch
mark('import torch: done')

mark('import lerobot dataset: start')
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
mark('import lerobot dataset: done')

mark('patch datasets cache: start')
import datasets.config as ds_config
hf_home = os.environ.get('HF_HOME')
cache_dir = os.environ.get('HF_DATASETS_CACHE') or (os.path.join(hf_home, 'datasets') if hf_home else None)
if cache_dir:
    os.makedirs(cache_dir, exist_ok=True)
    ds_config.HF_DATASETS_CACHE = cache_dir
    if hf_home:
        ds_config.HF_CACHE_HOME = hf_home
mark(f'patch datasets cache: done -> {getattr(ds_config, "HF_DATASETS_CACHE", None)}')

mark('import fsdp_value_sft_worker: start')
from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker
mark('import fsdp_value_sft_worker: done')

mark('import train_value module path exists: start')
train_value = '/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/rlinf-recap/examples/recap/value/train_value.py'
print(f'[probe] train_value_exists={os.path.exists(train_value)}', flush=True)
mark('done')
