#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-install}

case "$MODE" in
  install | --check) ;;
  *)
    echo "Usage: $0 [install|--check]" >&2
    exit 2
    ;;
esac

find_cuda() {
  if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    return
  fi

  local candidate
  for candidate in /usr/local/cuda-12.1 /usr/local/cuda-12.8 /usr/local/cuda; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      CUDA_HOME=$candidate
      export CUDA_HOME
      return
    fi
  done

  echo "CUDA toolkit not found. Set CUDA_HOME to a directory containing bin/nvcc." >&2
  exit 1
}

require_build_tools() {
  local missing=()
  local command
  for command in git gcc g++ make cmake ninja; do
    command -v "$command" >/dev/null 2>&1 || missing+=("$command")
  done
  if (( ${#missing[@]} )); then
    echo "Missing build tools: ${missing[*]}" >&2
    exit 1
  fi
}

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV=$(command -v uv)
    return
  fi

  for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [[ -x "$candidate" ]]; then
      UV=$candidate
      return
    fi
  done

  if [[ "$MODE" == "--check" ]]; then
    echo "uv is not installed." >&2
    exit 1
  fi

  command -v curl >/dev/null 2>&1 || {
    echo "curl is required to bootstrap uv." >&2
    exit 1
  }
  echo "Installing uv with the official installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV=${UV_INSTALL_DIR:-$HOME/.local/bin}/uv
  if [[ ! -x "$UV" ]]; then
    UV=$HOME/.cargo/bin/uv
  fi
  [[ -x "$UV" ]] || {
    echo "uv installation completed but the executable was not found." >&2
    exit 1
  }
}

configure_build_environment() {
  export CUDA_PATH="$CUDA_HOME"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
  export MAX_JOBS="${MAX_JOBS:-8}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
  export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
}

verify_environment() {
  local python=$ROOT/.venv/bin/python
  [[ -x "$python" ]] || {
    echo "Missing virtual environment: $ROOT/.venv" >&2
    exit 1
  }

  "$python" - <<'PY'
import importlib
import sys

required = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python-headless",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "msgspec": "msgspec",
    "numpy": "numpy",
    "pycolmap": "pycolmap",
    "romatch": "romatch",
    "romav2": "romav2",
    "gsplat": "gsplat",
}

failed = []
for module, package in required.items():
    try:
        importlib.import_module(module)
    except Exception as exc:
        failed.append(f"{package}: {exc}")

if failed:
    print("Environment verification failed:", file=sys.stderr)
    for error in failed:
        print(f"  {error}", file=sys.stderr)
    raise SystemExit(1)

import torch
import gsplat
from hydra import compose, initialize_config_dir

from ftgspp.config import from_hydra
from ftgspp.config.paths import PROJECT_ROOT
from ftgspp.config.register import register_configs

register_configs()
with initialize_config_dir(
    config_dir=str(PROJECT_ROOT / "configs"),
    version_base="1.3",
):
    composed = compose(
        config_name="config",
        overrides=["data=corgi_first300_scale05", "pipeline=check"],
    )
config = from_hydra(composed)
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gsplat={getattr(gsplat, '__version__', 'unknown')}")
print("hydra_compose=ok")
print(f"config_frames={config.data.frames.start}:{config.data.frames.stop}")
print(f"config_scale={config.data.scale}")
print(f"config_points={config.init.points_path}")
PY
}

cd "$ROOT"
find_cuda
require_build_tools
find_uv
configure_build_environment

echo "project=$ROOT"
echo "uv=$UV"
echo "cuda_home=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | tail -n 1

if [[ "$MODE" == "--check" ]]; then
  verify_environment
  exit 0
fi

"$UV" python install 3.12
"$UV" sync --frozen --python 3.12 --group with-torch

# gsplat is intentionally not declared in pyproject.toml because it is
# distributed under its own license. Keep this install after uv sync so a
# future environment reconciliation cannot remove it before verification.
"$UV" pip install --python "$ROOT/.venv/bin/python" "gsplat==1.5.3"

verify_environment
echo "FTGSPP environment is ready: $ROOT/.venv"
