#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PI05_STANDALONE_ENV_ROOT="${PI05_STANDALONE_ENV_ROOT:-${CONDA_PREFIX:-}}"
HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$REPO_ROOT/.cache/openpi}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$REPO_ROOT/.cache/lerobot}"
LEROBOT_LOCAL_HOME="${LEROBOT_LOCAL_HOME:-$HF_LEROBOT_HOME/local}"
PI05_LEROBOT_DATASET_DIR="${PI05_LEROBOT_DATASET_DIR:-$REPO_ROOT/data/lerobot_pick_place}"
PI05_LEROBOT_REPO_ID="${PI05_LEROBOT_REPO_ID:-pi05-pickplace-il}"

if [[ -z "$PI05_STANDALONE_ENV_ROOT" ]]; then
  echo "CONDA_PREFIX is empty. Activate your standalone conda env first." >&2
  echo "Example: conda activate pi05" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -x "$PI05_STANDALONE_ENV_ROOT/bin/python" ]]; then
  echo "Missing python at $PI05_STANDALONE_ENV_ROOT/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p "$HF_DATASETS_CACHE" "$MPLCONFIGDIR" "$OPENPI_DATA_HOME" "$LEROBOT_LOCAL_HOME"

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

_prepend_path PATH "$PI05_STANDALONE_ENV_ROOT/bin"
_prepend_path PYTHONPATH "$REPO_ROOT"
_prepend_path PYTHONPATH "$REPO_ROOT/openpi_official/src"
_prepend_path LD_LIBRARY_PATH "$PI05_STANDALONE_ENV_ROOT/lib"

if [[ -d "$PI05_LEROBOT_DATASET_DIR" ]]; then
  target_link="$LEROBOT_LOCAL_HOME/$PI05_LEROBOT_REPO_ID"
  if [[ -L "$target_link" ]]; then
    current_target="$(readlink "$target_link")"
    if [[ "$current_target" != "$PI05_LEROBOT_DATASET_DIR" ]]; then
      rm -f "$target_link"
      ln -s "$PI05_LEROBOT_DATASET_DIR" "$target_link"
    fi
  elif [[ ! -e "$target_link" ]]; then
    ln -s "$PI05_LEROBOT_DATASET_DIR" "$target_link"
  fi
fi

export PATH
export PYTHONPATH
export LD_LIBRARY_PATH
export HF_HOME
export HF_DATASETS_CACHE
export MPLCONFIGDIR
export OPENPI_DATA_HOME
export HF_LEROBOT_HOME
export LEROBOT_LOCAL_HOME
export PI05_LEROBOT_DATASET_DIR
export PI05_LEROBOT_REPO_ID
export OPENPI_LOCAL_PYTHON="$PI05_STANDALONE_ENV_ROOT/bin/python"
export OPENPI_LOCAL_REPO_ROOT="$REPO_ROOT"

echo "OPENPI_LOCAL_PYTHON=$OPENPI_LOCAL_PYTHON"
echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo "PI05_LEROBOT_DATASET_DIR=$PI05_LEROBOT_DATASET_DIR"
