#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$PI05_REPO_ROOT/openpi_official}"
PI05_CONDA_ENV_ROOT="${PI05_CONDA_ENV_ROOT:-$PI05_REPO_ROOT/.conda-pi05-openpi-final}"

LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-$OPENPI_ROOT/third_party/libero}"
LIBERO_PACKAGE_ROOT="${LIBERO_PACKAGE_ROOT:-$LIBERO_REPO_PATH/libero/libero}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$PI05_REPO_ROOT/.cache/libero}"
LIBERO_DATASETS_DIR="${LIBERO_DATASETS_DIR:-$PI05_REPO_ROOT/data/libero/datasets}"

HF_HOME="${HF_HOME:-$PI05_REPO_ROOT/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$PI05_REPO_ROOT/.cache/lerobot}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$PI05_REPO_ROOT/.cache/openpi}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$PI05_REPO_ROOT/.cache/matplotlib}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PI05_REPO_ROOT/.cache/numba}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PI05_REPO_ROOT/.cache/xdg}"

MUJOCO_GL="${MUJOCO_GL:-egl}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [[ ! -x "$PI05_CONDA_ENV_ROOT/bin/python" ]]; then
  echo "Missing conda python at $PI05_CONDA_ENV_ROOT/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "$LIBERO_REPO_PATH" ]]; then
  echo "Missing LIBERO repo at $LIBERO_REPO_PATH" >&2
  return 1 2>/dev/null || exit 1
fi

mkdir -p \
  "$LIBERO_CONFIG_PATH" \
  "$LIBERO_DATASETS_DIR" \
  "$HF_DATASETS_CACHE" \
  "$HF_LEROBOT_HOME" \
  "$OPENPI_DATA_HOME" \
  "$MPLCONFIGDIR" \
  "$NUMBA_CACHE_DIR" \
  "$XDG_CACHE_HOME"

cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
assets: $LIBERO_PACKAGE_ROOT/assets
bddl_files: $LIBERO_PACKAGE_ROOT/bddl_files
benchmark_root: $LIBERO_PACKAGE_ROOT
datasets: $LIBERO_DATASETS_DIR
init_states: $LIBERO_PACKAGE_ROOT/init_files
EOF

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

_prepend_path PATH "$PI05_CONDA_ENV_ROOT/bin"
_prepend_path PYTHONPATH "$OPENPI_ROOT/src"
_prepend_path PYTHONPATH "$OPENPI_ROOT/packages/openpi-client/src"
_prepend_path PYTHONPATH "$LIBERO_REPO_PATH"
_prepend_path PYTHONPATH "$PI05_REPO_ROOT"
_prepend_path LD_LIBRARY_PATH "$PI05_CONDA_ENV_ROOT/lib"

export PATH
export PYTHONPATH
export LD_LIBRARY_PATH
export PI05_REPO_ROOT
export OPENPI_ROOT
export PI05_CONDA_ENV_ROOT
export LIBERO_REPO_PATH
export LIBERO_CONFIG_PATH
export LIBERO_DATASETS_DIR
export HF_HOME
export HF_DATASETS_CACHE
export HF_LEROBOT_HOME
export OPENPI_DATA_HOME
export MPLCONFIGDIR
export NUMBA_CACHE_DIR
export XDG_CACHE_HOME
export MUJOCO_GL
export PYOPENGL_PLATFORM
export OPENPI_LOCAL_PYTHON="$PI05_CONDA_ENV_ROOT/bin/python"

echo "OPENPI_LOCAL_PYTHON=$OPENPI_LOCAL_PYTHON"
echo "OPENPI_ROOT=$OPENPI_ROOT"
echo "LIBERO_REPO_PATH=$LIBERO_REPO_PATH"
echo "LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH"
echo "LIBERO_DATASETS_DIR=$LIBERO_DATASETS_DIR"
echo "HF_HOME=$HF_HOME"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "MUJOCO_GL=$MUJOCO_GL"
