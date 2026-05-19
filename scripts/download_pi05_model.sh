#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR="${1:-/nfs_global/S/yangrongzheng/RLinf-main/models/RLinf-Pi05-SFT}"

source "$SCRIPT_DIR/use_local_openpi_env.sh" >/dev/null

if ! "$OPENPI_LOCAL_PYTHON" -c "import modelscope" >/dev/null 2>&1; then
  "$OPENPI_LOCAL_PYTHON" -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
fi

"$OPENPI_LOCAL_PYTHON" - <<PY
from modelscope.hub.file_download import model_file_download

model_id = "RLinf/RLinf-Pi05-SFT"
model_dir = r"$MODEL_DIR"

for file_path in [
    "README.md",
    "configuration.json",
    "physical-intelligence/libero/norm_stats.json",
    "model.safetensors",
]:
    print(f"downloading {file_path} -> {model_dir}", flush=True)
    saved_path = model_file_download(
        model_id=model_id,
        file_path=file_path,
        revision="master",
        local_dir=model_dir,
    )
    print(f"saved {saved_path}", flush=True)
PY
