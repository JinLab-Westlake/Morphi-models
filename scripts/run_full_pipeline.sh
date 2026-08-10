#!/bin/bash

set -euo pipefail


# ============================================================
# GPU
# ============================================================

GPU=0

export CUDA_VISIBLE_DEVICES=${GPU}

CELL_SEG_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# need input image and checkpoint
IMAGE_PATH="$1"


[ $# -eq 1 ] || { echo "Usage: $0  <image> <checkpoint>"; exit 1; }
# ============================================================
# Project roots
# ============================================================

MORPHI_ROOT="${CELL_SEG_ROOT}/Morphi"

DOANET_ROOT="${MORPHI_ROOT}/DoANet"

INSTANCE_ROOT="${MORPHI_ROOT}/Instance-segment"

PIPELINE_ROOT="${CELL_SEG_ROOT}/scripts"


# ============================================================
# Python executables
# ============================================================

DOANET_PYTHON="python"
INSTANCE_PYTHON="python"

# ============================================================
# Scripts
# ============================================================

DOANET_SCRIPT="${DOANET_ROOT}/inference_full-image.py"

CONVERT_SCRIPT="${PIPELINE_ROOT}/convert_semantic_mask.py"

INSTANCE_SCRIPT="${INSTANCE_ROOT}/tools/inference/inference_with_LIVE_MODEL.py"


## ============================================================
## Input
## ============================================================
#
#IMAGE_PATH="/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928/20240926_a2780_post_drug__2024-09-26T22_59_12-Measurement/channel/BF/r14c19f01.tiff"
#
#
## ============================================================
## DoANet model
## ============================================================
#
## DOANET_CHECKPOINT="${DOANET_ROOT}/output/inference.pth"
#DOANET_CHECKPOINT="/storage/jinxinLab/wangyimin/Cell_seg/Morphi-models/Morphi/DoANet/model_out20260730/epoch_264.pth"

DOANET_CHECKPOINT="${MORPHI_ROOT}/checkpoints/doanet/DoA.pth"

# DOANET_CFG="${DOANET_ROOT}/configs/your_config.yaml"
DOANET_CFG="configs/swin_tiny_patch4_window7_224_lite.yaml"

PATCH_SIZE=224

STRIDE=224

NUM_CLASSES=3


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

CASE_NAME="${IMAGE_STEM}"


# ============================================================
# Expected DoANet outputs
# ============================================================

DOANET_CASE_DIR="${DOANET_OUTPUT}/${IMAGE_STEM}"

RAW_PRED="${DOANET_CASE_DIR}/pred/${IMAGE_STEM}_pred_raw.png"


# ============================================================
# Temporary folders required by instance model
# ============================================================

INSTANCE_IMAGE_DIR="${INSTANCE_ROOT_A}/${CASE_NAME}/channel/bf"

INSTANCE_SEMANTIC_DIR="${INSTANCE_ROOT_B}/${CASE_NAME}/combined"

INSTANCE_RESULT_DIR="${INSTANCE_ROOT_A}/${CASE_NAME}/channel/output"


# ============================================================
# Converted semantic mask
# ============================================================

SEMANTIC_MASK="${INSTANCE_SEMANTIC_DIR}/${IMAGE_STEM}.png"


# ============================================================
# Final result
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
echo "Input image:"
echo "  ${IMAGE_PATH}"
echo ""
echo "DoANet script:"
echo "  ${DOANET_SCRIPT}"
echo ""
echo "Instance script:"
echo "  ${INSTANCE_SCRIPT}"
echo ""
echo "GPU:"
echo "  ${GPU}"
echo ""
echo "============================================================"
echo ""


# ============================================================
# Check required files
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

if [ ! -f "${CONVERT_SCRIPT}" ]; then
    echo "ERROR: conversion script not found:"
    echo "${CONVERT_SCRIPT}"
    exit 1
fi

if [ ! -f "${INSTANCE_SCRIPT}" ]; then
    echo "ERROR: instance script not found:"
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
    --img_size "${PATCH_SIZE}" \
    --stride "${STRIDE}" \
    --num_classes "${NUM_CLASSES}"


# ============================================================
# Check DoANet output
# ============================================================

if [ ! -f "${RAW_PRED}" ]; then
    echo ""
    echo "ERROR: DoANet prediction not found:"
    echo "${RAW_PRED}"
    exit 1
fi

echo ""
echo "DoANet prediction:"
echo "${RAW_PRED}"


# ============================================================
# STEP 2
# Prepare directories required by instance model
# ============================================================

echo ""
echo "============================================================"
echo "[2/4] Prepare instance segmentation input"
echo "============================================================"
echo ""

mkdir -p "${INSTANCE_IMAGE_DIR}"
mkdir -p "${INSTANCE_SEMANTIC_DIR}"

TARGET_IMAGE="${INSTANCE_IMAGE_DIR}/${IMAGE_FILENAME}"

rm -f "${TARGET_IMAGE}"

ln -s \
    "$(realpath "${IMAGE_PATH}")" \
    "${TARGET_IMAGE}"

echo "Instance image:"
echo "${TARGET_IMAGE}"


# ============================================================
# STEP 3
# Convert DoANet class IDs
#
# 0 -> 0
# 1 -> 127
# 2 -> 254
# ============================================================

echo ""
echo "============================================================"
echo "[3/4] Convert live/dead semantic mask"
echo "============================================================"
echo ""

"${DOANET_PYTHON}" "${CONVERT_SCRIPT}" \
    --input "${RAW_PRED}" \
    --output "${SEMANTIC_MASK}"

if [ ! -f "${SEMANTIC_MASK}" ]; then
    echo ""
    echo "ERROR: semantic mask was not generated:"
    echo "${SEMANTIC_MASK}"
    exit 1
fi

echo ""
echo "Semantic mask:"
echo "${SEMANTIC_MASK}"


# ============================================================
# STEP 4
# Instance segmentation + death rate
# ============================================================

echo ""
echo "============================================================"
echo "[4/4] Instance segmentation + organoid death rate"
echo "============================================================"
echo ""

INSTANCE_CONFIG="${INSTANCE_ROOT}/configs/organoids/config.yaml"
INSTANCE_CHECKPOINT="${MORPHI_ROOT}/checkpoints/instance/Organoid-seg.pth"

cd "${INSTANCE_ROOT}"

"${INSTANCE_PYTHON}" \
    "${INSTANCE_SCRIPT}" \
    "${INSTANCE_ROOT_A}" \
    "${INSTANCE_ROOT_B}" \
    "${INSTANCE_CONFIG}" \
    "${INSTANCE_CHECKPOINT}"


# ============================================================
# Check instance result
# ============================================================

if [ ! -d "${INSTANCE_RESULT_DIR}" ]; then
    echo ""
    echo "ERROR: instance output directory not found:"
    echo "${INSTANCE_RESULT_DIR}"
    exit 1
fi


# ============================================================
# Collect final results
# ============================================================

mkdir -p "${FINAL_CASE_DIR}"

cp -r \
    "${INSTANCE_RESULT_DIR}/." \
    "${FINAL_CASE_DIR}/"

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
echo "Final results:"
echo "  ${FINAL_CASE_DIR}"
echo ""
echo "Expected:"
echo "  ${FINAL_CASE_DIR}/instance_death_rate.xlsx"
echo "  ${FINAL_CASE_DIR}/instance_mask/"
echo "  ${FINAL_CASE_DIR}/overlay/"
echo "  ${FINAL_CASE_DIR}/${IMAGE_STEM}_semantic.png"
echo ""
echo "============================================================"