#!/bin/bash

set -euo pipefail


# ============================================================
# GPU
# ============================================================

GPU=0
export CUDA_VISIBLE_DEVICES=${GPU}


# ============================================================
# Main project paths
# ============================================================

CELL_SEG_ROOT="/storage/jinxinLab/wangyimin/Cell_seg"

DOANET_ROOT="${CELL_SEG_ROOT}/DoANet"

INSTANCE_ROOT="${CELL_SEG_ROOT}/Instance-segment"


# ============================================================
# Python executables
#
# 如果两个项目使用同一个conda环境：
#
# DOANET_PYTHON="python"
# INSTANCE_PYTHON="python"
#
# 如果是两个环境，就分别写绝对路径。
# ============================================================

DOANET_PYTHON="python"
INSTANCE_PYTHON="python"


# ============================================================
# Input image
# ============================================================

IMAGE_PATH="/storage/jinxinLab/wangyimin/Cell_seg/Datasets/test/test_1080.png"


# ============================================================
# DoANet settings
# ============================================================

DOANET_SCRIPT="${DOANET_ROOT}/inference_single_image.py"

DOANET_CHECKPOINT="${DOANET_ROOT}/output/inference.pth"

DOANET_CFG="${DOANET_ROOT}/configs/your_config.yaml"

PATCH_SIZE=224
STRIDE=224
NUM_CLASSES=3


# ============================================================
# Instance segmentation script
# ============================================================

INSTANCE_SCRIPT="${INSTANCE_ROOT}/tools/inference/inference_with_LIVE_MODEL.py"


# ============================================================
# Pipeline workspace
# ============================================================

WORK_ROOT="${CELL_SEG_ROOT}/pipeline_tmp"

DOANET_OUTPUT="${WORK_ROOT}/doanet"

INSTANCE_ROOT_A="${WORK_ROOT}/instance_input"

INSTANCE_ROOT_B="${WORK_ROOT}/semantic_input"

FINAL_OUTPUT="${CELL_SEG_ROOT}/pipeline_results"


# ============================================================
# Image information
# ============================================================

IMAGE_FILENAME=$(basename "${IMAGE_PATH}")

IMAGE_STEM="${IMAGE_FILENAME%.*}"

# 用图像名字作为临时 case name
CASE_NAME="${IMAGE_STEM}"


# ============================================================
# Expected DoANet output
# ============================================================

DOANET_CASE_DIR="${DOANET_OUTPUT}/${IMAGE_STEM}"

RAW_PRED="${DOANET_CASE_DIR}/pred/${IMAGE_STEM}_pred_raw.png"


# ============================================================
# Directory structure required by inference_with_LIVE_MODEL.py
# ============================================================

INSTANCE_IMAGE_DIR="${INSTANCE_ROOT_A}/${CASE_NAME}/channel/more_bf"

INSTANCE_SEMANTIC_DIR="${INSTANCE_ROOT_B}/${CASE_NAME}/combined"

INSTANCE_RESULT_DIR="${INSTANCE_ROOT_A}/${CASE_NAME}/channel/output_r02c02"


# ============================================================
# Final output directory
# ============================================================

FINAL_CASE_DIR="${FINAL_OUTPUT}/${CASE_NAME}"


# ============================================================
# Print configuration
# ============================================================

echo ""
echo "============================================================"
echo "     DoANet + Instance Segmentation Pipeline"
echo "============================================================"
echo ""
echo "Input:"
echo "  ${IMAGE_PATH}"
echo ""
echo "DoANet:"
echo "  ${DOANET_SCRIPT}"
echo "  ${DOANET_CHECKPOINT}"
echo ""
echo "Instance:"
echo "  ${INSTANCE_SCRIPT}"
echo ""
echo "GPU:"
echo "  ${GPU}"
echo ""
echo "Case:"
echo "  ${CASE_NAME}"
echo ""
echo "============================================================"
echo ""


# ============================================================
# Check files
# ============================================================

if [ ! -f "${IMAGE_PATH}" ]; then
    echo "ERROR: image not found:"
    echo "${IMAGE_PATH}"
    exit 1
fi


if [ ! -f "${DOANET_SCRIPT}" ]; then
    echo "ERROR: DoANet script not found:"
    echo "${DOANET_SCRIPT}"
    exit 1
fi


if [ ! -f "${DOANET_CHECKPOINT}" ]; then
    echo "ERROR: DoANet checkpoint not found:"
    echo "${DOANET_CHECKPOINT}"
    exit 1
fi


if [ ! -f "${INSTANCE_SCRIPT}" ]; then
    echo "ERROR: Instance script not found:"
    echo "${INSTANCE_SCRIPT}"
    exit 1
fi


# ============================================================
# STEP 1
# DoANet semantic segmentation
# ============================================================

echo ""
echo "============================================================"
echo "[1/4] DoANet semantic segmentation"
echo "============================================================"
echo ""


mkdir -p "${DOANET_OUTPUT}"


cd "${DOANET_ROOT}"


"${DOANET_PYTHON}" "${DOANET_SCRIPT}" \
    --image_path "${IMAGE_PATH}" \
    --output_dir "$(dirname "${DOANET_CHECKPOINT}")" \
    --checkpoint "${DOANET_CHECKPOINT}" \
    --test_save_dir "${DOANET_OUTPUT}" \
    --cfg "${DOANET_CFG}" \
    --img_size ${PATCH_SIZE} \
    --stride ${STRIDE} \
    --num_classes ${NUM_CLASSES}


# ============================================================
# Check DoANet output
# ============================================================

if [ ! -f "${RAW_PRED}" ]; then

    echo ""
    echo "ERROR: DoANet raw prediction not found:"
    echo "${RAW_PRED}"
    echo ""

    exit 1
fi


echo ""
echo "DoANet raw prediction:"
echo "${RAW_PRED}"


# ============================================================
# STEP 2
# Prepare folder structure required by instance model
# ============================================================

echo ""
echo "============================================================"
echo "[2/4] Prepare instance pipeline input"
echo "============================================================"
echo ""


mkdir -p "${INSTANCE_IMAGE_DIR}"

mkdir -p "${INSTANCE_SEMANTIC_DIR}"


# ------------------------------------------------------------
# Put original image into:
#
# root_a/case/channel/more_bf/
#
# 使用软链接避免复制1080图
# ------------------------------------------------------------

TARGET_IMAGE="${INSTANCE_IMAGE_DIR}/${IMAGE_FILENAME}"


rm -f "${TARGET_IMAGE}"


ln -s \
    "$(realpath "${IMAGE_PATH}")" \
    "${TARGET_IMAGE}"


echo "Instance input image:"
echo "${TARGET_IMAGE}"


# ============================================================
# STEP 3
# Convert DoANet labels:
#
# raw:
#
#   0 = background
#   1 = live
#   2 = dead
#
# instance script expects:
#
#   0   = background
#   127 = live
#   254 = dead
#
# ============================================================

echo ""
echo "============================================================"
echo "[3/4] Prepare live/dead semantic mask"
echo "============================================================"
echo ""


SEMANTIC_MASK="${INSTANCE_SEMANTIC_DIR}/${IMAGE_STEM}.png"


"${DOANET_PYTHON}" - \
    "${RAW_PRED}" \
    "${SEMANTIC_MASK}" <<'PY'

import sys
import numpy as np
from PIL import Image


src = sys.argv[1]
dst = sys.argv[2]


mask = np.array(
    Image.open(src)
)


if mask.ndim == 3:
    mask = mask[:, :, 0]


unique = np.unique(mask)

print(
    "Raw DoANet labels:",
    unique
)


allowed = {0, 1, 2}

actual = set(
    unique.tolist()
)


if not actual.issubset(allowed):

    raise RuntimeError(
        f"Unexpected DoANet labels: {unique}. "
        "Expected only 0, 1, 2."
    )


out = np.zeros(
    mask.shape,
    dtype=np.uint8
)


out[mask == 0] = 0
out[mask == 1] = 127
out[mask == 2] = 254


Image.fromarray(
    out
).save(
    dst
)


print(
    "Converted semantic labels:",
    np.unique(out)
)

print(
    "Saved semantic mask:",
    dst
)

PY


if [ ! -f "${SEMANTIC_MASK}" ]; then

    echo ""
    echo "ERROR: semantic mask was not generated."
    echo ""

    exit 1
fi


# ============================================================
# At this point:
#
# root_a:
#
# instance_input/
# └── test_1080/
#     └── channel/
#         └── more_bf/
#             └── test_1080.png
#
#
# root_b:
#
# semantic_input/
# └── test_1080/
#     └── combined/
#         └── test_1080.png
#
#
# Exactly matches inference_with_LIVE_MODEL.py.
# ============================================================


echo ""
echo "Instance ROOT_A:"
echo "${INSTANCE_ROOT_A}"

echo ""
echo "Instance ROOT_B:"
echo "${INSTANCE_ROOT_B}"


# ============================================================
# STEP 4
# Instance segmentation + death rate
# ============================================================

echo ""
echo "============================================================"
echo "[4/4] Instance segmentation + organoid death rate"
echo "============================================================"
echo ""


cd "${INSTANCE_ROOT}"


"${INSTANCE_PYTHON}" \
    "${INSTANCE_SCRIPT}" \
    "${INSTANCE_ROOT_A}" \
    "${INSTANCE_ROOT_B}"


# ============================================================
# Check result
# ============================================================

if [ ! -d "${INSTANCE_RESULT_DIR}" ]; then

    echo ""
    echo "ERROR: Instance output directory not found:"
    echo "${INSTANCE_RESULT_DIR}"

    exit 1
fi


# ============================================================
# Copy final results
# ============================================================

mkdir -p "${FINAL_CASE_DIR}"


cp -r \
    "${INSTANCE_RESULT_DIR}/." \
    "${FINAL_CASE_DIR}/"


# Also keep semantic mask
cp \
    "${SEMANTIC_MASK}" \
    "${FINAL_CASE_DIR}/${IMAGE_STEM}_semantic.png"


# ============================================================
# Finished
# ============================================================

echo ""
echo "============================================================"
echo "                    FINISHED"
echo "============================================================"
echo ""
echo "Final output:"
echo ""
echo "  ${FINAL_CASE_DIR}"
echo ""
echo "Expected files:"
echo ""
echo "  ${FINAL_CASE_DIR}/instance_death_rate.xlsx"
echo ""
echo "  ${FINAL_CASE_DIR}/instance_mask/"
echo ""
echo "  ${FINAL_CASE_DIR}/overlay/"
echo ""
echo "  ${FINAL_CASE_DIR}/${IMAGE_STEM}_semantic.png"
echo ""
echo "============================================================"