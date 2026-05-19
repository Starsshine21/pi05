#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/use_local_openpi_env.sh"

"$OPENPI_LOCAL_PYTHON" "$SCRIPT_DIR/check_local_openpi_env.py"
