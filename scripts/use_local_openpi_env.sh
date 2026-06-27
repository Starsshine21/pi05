#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PI05_CONDA_ENV_ROOT="${PI05_CONDA_ENV_ROOT:-$REPO_ROOT/.conda-pi05-openpi-final}"
RECAP_WORKSPACE="${RECAP_WORKSPACE:-$REPO_ROOT/recap_workspace}"
RLINF_ROOT="${RLINF_ROOT:-$RECAP_WORKSPACE/vendor/rlinf-recap}"
RLINF_VENV_ROOT="${RLINF_VENV_ROOT:-$RECAP_WORKSPACE/.venv_recap311}"
RLINF_VENV_SITE_PACKAGES="${RLINF_VENV_SITE_PACKAGES:-$RLINF_VENV_ROOT/lib/python3.11/site-packages}"
LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-$RECAP_WORKSPACE/vendor/libero}"
CUDA_RUNTIME_ROOT="${CUDA_RUNTIME_ROOT:-$RLINF_VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/data}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/.cache/openpi}"

if [[ ! -x "$PI05_CONDA_ENV_ROOT/bin/python" ]]; then
  echo "Missing conda python at $PI05_CONDA_ENV_ROOT/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -x "$RLINF_VENV_ROOT/bin/python" ]]; then
  export OPENPI_LOCAL_PYTHON="$RLINF_VENV_ROOT/bin/python"
else
  export OPENPI_LOCAL_PYTHON="$PI05_CONDA_ENV_ROOT/bin/python"
fi

if [[ ! -d "$RLINF_VENV_SITE_PACKAGES" ]]; then
  echo "Missing RLinf venv site-packages at $RLINF_VENV_SITE_PACKAGES" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$HF_DATASETS_CACHE" "$MPLCONFIGDIR" "$OPENPI_DATA_HOME"

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

_remove_path_segment() {
  local var_name="$1"
  local remove_value="$2"
  local current="${!var_name:-}"
  local rebuilt=""
  local segment
  IFS=':' read -r -a _segments <<< "$current"
  for segment in "${_segments[@]}"; do
    [[ -z "$segment" ]] && continue
    [[ "$segment" == "$remove_value" ]] && continue
    if [[ -z "$rebuilt" ]]; then
      rebuilt="$segment"
    else
      rebuilt="$rebuilt:$segment"
    fi
  done
  printf -v "$var_name" '%s' "$rebuilt"
}

_remove_path_segment PYTHONPATH "/nfs_global/S/yangrongzheng/RLinf-main"
_remove_path_segment PYTHONPATH "/nfs_global/S/yangrongzheng/RLinf-main/.venv/lib/python3.11/site-packages"
_remove_path_segment PYTHONPATH "/nfs_global/S/yangrongzheng/RLinf-main/RLinf_deps/libero"
_remove_path_segment PATH "/nfs_global/S/yangrongzheng/RLinf-main/.venv/bin"

declare -a LD_SEGMENTS=(
  "$PI05_CONDA_ENV_ROOT/lib"
)

while IFS= read -r lib_dir; do
  LD_SEGMENTS+=("$lib_dir")
done < <(
  find "$RLINF_VENV_SITE_PACKAGES" -type d \
    \( -path "*/nvidia/*/lib" -o -path "*/cusparselt/lib" \) | sort
)

_prepend_path PATH "$RLINF_VENV_ROOT/bin"
_prepend_path PATH "$PI05_CONDA_ENV_ROOT/bin"
_prepend_path PYTHONPATH "$REPO_ROOT/openpi_official/src"
_prepend_path PYTHONPATH "$REPO_ROOT"
_prepend_path PYTHONPATH "$RLINF_ROOT"
_prepend_path PYTHONPATH "$LIBERO_REPO_PATH"
_prepend_path PYTHONPATH "$RLINF_VENV_SITE_PACKAGES"

for segment in "${LD_SEGMENTS[@]}"; do
  _prepend_path LD_LIBRARY_PATH "$segment"
done

export PATH
export PYTHONPATH
export LD_LIBRARY_PATH
export CUDA_RUNTIME_ROOT
export PI05_CONDA_ENV_ROOT
export RECAP_WORKSPACE
export RLINF_ROOT
export RLINF_VENV_ROOT
export RLINF_VENV_SITE_PACKAGES
export LIBERO_REPO_PATH
export HF_LEROBOT_HOME
export HF_HOME
export HF_DATASETS_CACHE
export MPLCONFIGDIR
export OPENPI_DATA_HOME
export OPENPI_LOCAL_REPO_ROOT="$REPO_ROOT"

unset PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE
export PATH="$RLINF_VENV_ROOT/bin:$PI05_CONDA_ENV_ROOT/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

echo "OPENPI_LOCAL_PYTHON=$OPENPI_LOCAL_PYTHON"
echo "RLINF_VENV_SITE_PACKAGES=$RLINF_VENV_SITE_PACKAGES"
echo "CUDA_RUNTIME_ROOT=$CUDA_RUNTIME_ROOT"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "PATH=$PATH"
echo "PYTHONPATH=$PYTHONPATH"
