#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="morphi"

echo "=========================================="
echo "Installing Morphi"
echo "=========================================="

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available."
    exit 1
fi

echo "[1/4] Creating conda environment..."
conda env create \
    -f "${ROOT}/Morphi/env/environment.yaml"

echo "[2/4] Installing Detectron2..."
conda run -n "${ENV_NAME}" \
    python -m pip install -e "${ROOT}/Morphi/env/detectron2"

echo "[3/4] Installing Morphi..."
conda run -n "${ENV_NAME}" \
    python -m pip install -e "${ROOT}"

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

echo "=========================================="
echo "Morphi installation completed."
echo "=========================================="
echo "Activate with: conda activate ${ENV_NAME}"