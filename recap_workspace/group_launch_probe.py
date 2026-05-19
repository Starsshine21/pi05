import os
import sys
import time

from omegaconf import OmegaConf


def mark(msg: str) -> None:
    print(f"[group-probe] {time.strftime('%F %T')} {msg}", flush=True)

hf_home = '/nfs_global/S/yangrongzheng/pi05/.cache/huggingface'
os.environ.setdefault('HF_HOME', hf_home)
os.environ.setdefault('HF_DATASETS_CACHE', os.path.join(hf_home, 'datasets'))
os.environ.setdefault('HF_LEROBOT_HOME', '/nfs_global/S/yangrongzheng/pi05/.cache/huggingface/lerobot')
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(hf_home, 'transformers'))
os.environ.setdefault('REPO_PATH', '/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/rlinf-recap')
os.environ.setdefault('RAY_ADDRESS', 'r8l40s-a05:26490')

mark('imports start')
from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker
mark('imports done')

cfg = OmegaConf.load('/nfs_global/S/yangrongzheng/pi05/recap_workspace/configs/local_value_sft.yaml')
cfg = validate_cfg(cfg)
mark('validate_cfg done')
cluster = Cluster(cluster_cfg=cfg.cluster)
mark('cluster done')
component_placement = HybridComponentPlacement(cfg, cluster)
mark('placement object done')
actor_placement = component_placement.get_strategy('actor')
mark('get_strategy done')
actor_group = FSDPValueSftWorker.create_group(cfg).launch(
    cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
)
mark('launch done')
print(actor_group, flush=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
