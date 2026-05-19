import os
import runpy
import sys
import traceback

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _patch_hf_datasets_cache() -> None:
    hf_home = os.environ.get("HF_HOME")
    cache_dir = os.environ.get("HF_DATASETS_CACHE") or (os.path.join(hf_home, "datasets") if hf_home else None)
    if not cache_dir:
        return

    os.makedirs(cache_dir, exist_ok=True)

    import datasets.config as ds_config

    ds_config.HF_DATASETS_CACHE = cache_dir
    if hf_home:
        ds_config.HF_CACHE_HOME = hf_home




def _patch_lerobot_timestamp_sync() -> None:
    from lerobot.common.datasets import utils as lerobot_utils
    from lerobot.common.datasets import lerobot_dataset as lerobot_dataset_mod

    if getattr(lerobot_utils, '_recap_timestamp_patch_applied', False):
        return

    original = lerobot_utils.check_timestamps_sync

    def patched_check_timestamps_sync(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except ValueError as exc:
            print(f"[recap-runner] timestamp sync warning ignored: {exc}", flush=True)
            return None

    lerobot_utils.check_timestamps_sync = patched_check_timestamps_sync
    lerobot_dataset_mod.check_timestamps_sync = patched_check_timestamps_sync
    lerobot_utils._recap_timestamp_patch_applied = True

def _patch_lerobot_query_hf_dataset() -> None:
    if getattr(LeRobotDataset, '_recap_query_patch_applied', False):
        return

    def _query_hf_dataset_compat(self, query_indices):
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            values = self.hf_dataset.select(q_idx)[key]
            if hasattr(values, 'to_pylist'):
                values = values.to_pylist()
            else:
                values = list(values)
            if values and isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result

    LeRobotDataset._query_hf_dataset = _query_hf_dataset_compat
    LeRobotDataset._recap_query_patch_applied = True


def _patch_worker_init() -> None:
    from rlinf.workers.sft.fsdp_value_sft_worker import FSDPValueSftWorker

    if getattr(FSDPValueSftWorker, '_recap_init_patch_applied', False):
        return

    orig_init = FSDPValueSftWorker.__init__

    def patched_init(self, cfg):
        _patch_lerobot_query_hf_dataset()
        return orig_init(self, cfg)

    FSDPValueSftWorker.__init__ = patched_init
    FSDPValueSftWorker._recap_init_patch_applied = True


def _flush_and_exit(code: int) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


def main() -> None:
    print('[recap-runner] main: patch_hf_datasets_cache')
    _patch_hf_datasets_cache()
    print('[recap-runner] main: patch_lerobot_timestamp_sync')
    _patch_lerobot_timestamp_sync()
    print('[recap-runner] main: patch_lerobot_query_hf_dataset')
    _patch_lerobot_query_hf_dataset()
    print('[recap-runner] main: patch_worker_init')
    _patch_worker_init()
    train_value = '/nfs_global/S/yangrongzheng/pi05/recap_workspace/external/rlinf-recap/examples/recap/value/train_value.py'
    sys.argv = [train_value, *sys.argv[1:]]
    print(f'[recap-runner] main: run {train_value}')
    runpy.run_path(train_value, run_name='__main__')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        _flush_and_exit(1)
    else:
        print('[recap-runner] main: completed successfully')
        _flush_and_exit(0)
