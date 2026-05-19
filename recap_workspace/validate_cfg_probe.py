import json
import os
import time

from omegaconf import OmegaConf


def mark(msg: str) -> None:
    print(f"[validate-probe] {time.strftime('%F %T')} {msg}", flush=True)

hf_home = '/nfs_global/S/yangrongzheng/pi05/.cache/huggingface'
os.environ.setdefault('HF_HOME', hf_home)
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(hf_home, 'datasets'))
os.environ.setdefault('HF_LEROBOT_HOME', '/nfs_global/S/yangrongzheng/pi05/.cache/huggingface/lerobot')
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(hf_home, 'transformers'))
os.environ.setdefault('REPO_PATH', '/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/rlinf-recap')

mark('import hydra deps start')
from rlinf.config import validate_cfg
mark('import hydra deps done')

mark('load config start')
cfg = OmegaConf.load('/nfs_global/S/yangrongzheng/pi05/recap_workspace/configs/local_value_sft.yaml')
mark('load config done')

mark('resolve config start')
resolved = OmegaConf.to_container(cfg, resolve=True)
print(json.dumps(resolved, indent=2)[:2000], flush=True)
mark('resolve config done')

mark('validate_cfg start')
validated = validate_cfg(cfg)
mark('validate_cfg done')
print(type(validated), flush=True)
