#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RECAP_WORKSPACE="${RECAP_WORKSPACE:-$REPO_ROOT/recap_workspace}"
RLINF_ROOT="${RLINF_ROOT:-$RECAP_WORKSPACE/vendor/rlinf-recap}"
RLINF_VENV_ROOT="${RLINF_VENV_ROOT:-$RECAP_WORKSPACE/.venv_recap311}"
RLINF_VENV_SITE_PACKAGES="${RLINF_VENV_SITE_PACKAGES:-$RLINF_VENV_ROOT/lib/python3.11/site-packages}"
LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-$RECAP_WORKSPACE/vendor/libero}"
CUDA_RUNTIME_ROOT="${CUDA_RUNTIME_ROOT:-$RLINF_VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib}"
PI05_CONDA_ENV_ROOT="${PI05_CONDA_ENV_ROOT:-${CONDA_PREFIX:-$REPO_ROOT/.conda-pi05-openpi-final}}"

export _PI05_OLD_PYTHONPATH="${PYTHONPATH-}"
export _PI05_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export _PI05_OLD_CUDA_RUNTIME_ROOT="${CUDA_RUNTIME_ROOT-}"
export _PI05_OLD_PI05_CONDA_ENV_ROOT="${PI05_CONDA_ENV_ROOT-}"
export _PI05_OLD_RECAP_WORKSPACE="${RECAP_WORKSPACE-}"
export _PI05_OLD_RLINF_ROOT="${RLINF_ROOT-}"
export _PI05_OLD_RLINF_VENV_ROOT="${RLINF_VENV_ROOT-}"
export _PI05_OLD_RLINF_VENV_SITE_PACKAGES="${RLINF_VENV_SITE_PACKAGES-}"
export _PI05_OLD_LIBERO_REPO_PATH="${LIBERO_REPO_PATH-}"
export _PI05_OLD_OPENPI_LOCAL_PYTHON="${OPENPI_LOCAL_PYTHON-}"
export _PI05_OLD_OPENPI_LOCAL_REPO_ROOT="${OPENPI_LOCAL_REPO_ROOT-}"

_prepend_path() {
  local var_name="$1"
  local path_value="$2"
  local current="${!var_name:-}"
  [[ -e "$path_value" ]] || return 0

  case ":$current:" in
    *":$path_value:"*) return 0 ;;
  esac

  if [[ -z "$current" ]]; then
    printf -v "$var_name" '%s' "$path_value"
  else
    printf -v "$var_name" '%s:%s' "$path_value" "$current"
  fi
}

declare -a LD_SEGMENTS=(
  "$PI05_CONDA_ENV_ROOT/lib"
)

while IFS= read -r lib_dir; do
  LD_SEGMENTS+=("$lib_dir")
done < <(
  find "$RLINF_VENV_SITE_PACKAGES" -type d \
    \( -path "*/nvidia/*/lib" -o -path "*/cusparselt/lib" \) | sort
)

_prepend_path PYTHONPATH "$REPO_ROOT"
_prepend_path PYTHONPATH "$RLINF_ROOT"
_prepend_path PYTHONPATH "$RLINF_VENV_SITE_PACKAGES"
_prepend_path PYTHONPATH "$LIBERO_REPO_PATH"

for segment in "${LD_SEGMENTS[@]}"; do
  _prepend_path LD_LIBRARY_PATH "$segment"
done

export PYTHONPATH
export LD_LIBRARY_PATH
export CUDA_RUNTIME_ROOT
export PI05_CONDA_ENV_ROOT
export RECAP_WORKSPACE
export RLINF_ROOT
export RLINF_VENV_ROOT
export RLINF_VENV_SITE_PACKAGES
export LIBERO_REPO_PATH
export OPENPI_LOCAL_PYTHON="$PI05_CONDA_ENV_ROOT/bin/python"
export OPENPI_LOCAL_REPO_ROOT="$REPO_ROOT"
