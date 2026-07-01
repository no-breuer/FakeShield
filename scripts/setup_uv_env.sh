#!/usr/bin/env bash
# Create a uv-managed Python 3.9 environment for FakeShield on the GPU cluster.
#
# Pinned per the repo README:
#   Python 3.9, PyTorch 1.13.0, CUDA 11.6, transformers 4.28.0 (MFLM) / 4.37.2 (DTE-FDM),
#   mmcv v1.4.7 (built from source), flash-attn 2.3.6.
#
# test.sh re-pins transformers between stages (4.37.2 -> 4.28.0), so this env
# only needs the base 4.28.0; test.sh swaps it at runtime.
#
# Usage:
#   bash scripts/setup_uv_env.sh
# Override the PyTorch CUDA build via TORCH_INDEX, e.g. for CUDA 11.7:
#   TORCH_INDEX=cu117 bash scripts/setup_uv_env.sh
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX="${TORCH_INDEX:-cu116}"
MMCV_TAG="v1.4.7"

# --------------------------------------------------------------------------- #
# 0. uv venv (seed so `pip` is available for test.sh's re-pins)
#    --python-preference only-managed: force uv to download its own CPython
#    (python-build-standalone), which ships Python.h in its include dir.
#    The system python3.9 lacks dev headers (python3-devel), so C-extension
#    builds like pycocotools fail with 'Python.h: No such file or directory'.
# --------------------------------------------------------------------------- #
uv venv --python 3.9 --python-preference only-managed --seed "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python --version
uv --version

# --------------------------------------------------------------------------- #
# 1. PyTorch 1.13.0 + torchvision 0.14.0 (CUDA 11.6 wheels)
# --------------------------------------------------------------------------- #
echo "==== Installing torch 1.13.0+${TORCH_INDEX} ===="
uv pip install torch==1.13.0 torchvision==0.14.0 \
    --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

# --------------------------------------------------------------------------- #
# 2. Python deps from requirements.txt (transformers stays at 4.28.0 here)
# --------------------------------------------------------------------------- #
echo "==== Installing requirements.txt (excluding flash-attn; installed separately below) ===="
# flash-attn needs torch at build time but doesn't declare it, so it must be
# built with --no-build-isolation after torch is present (step 5).
grep -v '^flash-attn' requirements.txt | uv pip install -r -

# --------------------------------------------------------------------------- #
# 3. mmcv v1.4.7 from source (needs torch present -> no build isolation)
# --------------------------------------------------------------------------- #
echo "==== Installing mmcv ${MMCV_TAG} from source ===="
MMCV_SRC="$(mktemp -d)/mmcv"
git clone --depth 1 --branch "${MMCV_TAG}" https://github.com/open-mmlab/mmcv "${MMCV_SRC}"
MMCV_WITH_OPS=1 uv pip install -e "${MMCV_SRC}" --no-build-isolation

# --------------------------------------------------------------------------- #
# 4. DTE-FDM editable (--no-deps: requirements.txt already covers its deps,
#    avoids a scikit-learn 1.2.2 downgrade that conflicts with requirements.txt)
# --------------------------------------------------------------------------- #
echo "==== Installing DTE-FDM (editable, no-deps) ===="
uv pip install -e ./DTE-FDM --no-deps

# --------------------------------------------------------------------------- #
# 5. flash-attn 2.3.6 (needs torch + ninja + nvcc -> no build isolation)
#    flash-attn compiles CUDA kernels, so it needs nvcc on PATH and CUDA_HOME
#    set. On SLURM/HPC clusters, CUDA is loaded via the module system.
# --------------------------------------------------------------------------- #
echo "==== Ensuring nvcc / CUDA_HOME for flash-attn build ===="

# Try to load a CUDA module if nvcc isn't available yet.
if ! command -v nvcc >/dev/null 2>&1; then
    # Source the module system if it's available.
    if [ -f /etc/profile.d/modules.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh
    fi
    if command -v module >/dev/null 2>&1; then
        # Try common CUDA module names; first match wins.
        for mod in cuda/11.6 cuda/11.7 cuda/11.8 cuda/12.1 cuda cuda-toolkit; do
            if module avail -t 2>&1 | grep -q "^${mod}"; then
                echo "  loading module: ${mod}"
                module load "${mod}"
                break
            fi
        done
    fi
fi

# Derive CUDA_HOME from nvcc if still unset.
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. flash-attn cannot build without it." >&2
    echo "Load CUDA manually and re-run just this step:" >&2
    echo "  module avail cuda" >&2
    echo "  module load cuda/11.6   # (or whatever version is available)" >&2
    echo "  export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))" >&2
    echo "  uv pip install flash-attn==2.3.6 --no-build-isolation" >&2
    exit 1
fi

echo "  nvcc:       $(command -v nvcc)"
echo "  CUDA_HOME:  ${CUDA_HOME}"

echo "==== Installing flash-attn 2.3.6 ===="
MAX_JOBS=4 uv pip install flash-attn==2.3.6 --no-build-isolation

# --------------------------------------------------------------------------- #
# 6. Verify
# --------------------------------------------------------------------------- #
echo
echo "==== Verify ===="
python - <<'PY'
import torch, transformers, flash_attn, mmcv
print("torch        :", torch.__version__, "cuda?", torch.version.cuda, "avail", torch.cuda.is_available())
print("transformers :", transformers.__version__)
print("flash_attn   :", flash_attn.__version__)
print("mmcv         :", mmcv.__version__)
PY

echo
echo "==== setup_uv_env.sh finished ===="
echo "Activate later with:  source ${VENV_DIR}/bin/activate"
echo "Then run:             WEIGHT_PATH=/share/nils.breuer/weights/fakeshield-v1-22b bash scripts/test.sh"
