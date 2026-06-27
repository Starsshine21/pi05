import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="physical-intelligence/libero",
    repo_type="dataset",
    local_dir="/nfs_global/S/yangrongzheng/pi05/data/hf_datasets/physical-intelligence/libero",
    local_dir_use_symlinks=False
)