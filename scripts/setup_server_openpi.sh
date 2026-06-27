#!/usr/bin/env bash
set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-$PWD/openpi}"
OPENPI_REF="${OPENPI_REF:-c23745b5ad24e98f66967ea795a07b2588ed6c79}"
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_PARENT="$(cd "$(dirname "$OPENPI_DIR")" && pwd)"
OPENPI_DIR="$OPENPI_PARENT/$(basename "$OPENPI_DIR")"

if [ ! -d "$OPENPI_DIR/.git" ]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi "$OPENPI_DIR"
fi

cd "$OPENPI_DIR"
git fetch origin
git checkout "$OPENPI_REF"
git submodule update --init --recursive

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

python "$KIT_DIR/scripts/patch_openpi_advantage_prompt.py" "$OPENPI_DIR"

echo "openpi is ready at $OPENPI_DIR"
