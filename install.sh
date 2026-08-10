#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Installing Morphi"
echo "=========================================="

# ------------------------------------------------------------
# 1. Check conda
# ------------------------------------------------------------

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available."
    echo "Please install Miniconda or Anaconda first."
    exit 1
fi


# ------------------------------------------------------------
# 2. Create environment
# ------------------------------------------------------------

echo ""
echo "[1/4] Creating conda environment..."

conda env create \
    -f "${ROOT}/env/environment.yaml"


# ------------------------------------------------------------
# IMPORTANT:
# Replace Jupyter below with the actual `name:` value
# inside environment.yaml
# ------------------------------------------------------------

ENV_NAME="Jupyter"


# ------------------------------------------------------------
# 3. Install bundled Detectron2
# ------------------------------------------------------------

echo ""
echo "[2/4] Installing Detectron2..."

conda run -n "${ENV_NAME}" \
    python -m pip install -e "${ROOT}/env/detectron2"


# ------------------------------------------------------------
# 4. Install Morphi
# ------------------------------------------------------------

echo ""
echo "[3/4] Installing Morphi..."

conda run -n "${ENV_NAME}" \
    python -m pip install -e "${ROOT}"


# ------------------------------------------------------------
# 5. Verify
# ------------------------------------------------------------

echo ""
echo "[4/4] Verifying installation..."

conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import torchvision
import detectron2

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Detectron2:", detectron2.__version__)
print("Detectron2 path:", detectron2.__file__)

print("\nInstallation check passed.")
PY


echo ""
echo "=========================================="
echo "Morphi installation completed."
echo "=========================================="
echo ""
echo "Activate the environment with:"
echo "  conda activate ${ENV_NAME}"