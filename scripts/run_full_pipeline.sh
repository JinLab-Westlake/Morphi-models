#!/bin/bash

set -euo pipefail


# ============================================================
# GPU
# ============================================================

GPU=0

export CUDA_VISIBLE_DEVICES=${GPU}


# ============================================================
# Project roots
# ============================================================

CELL_SEG_ROOT="/storage/jinxinLab/wangyimin/Cell_seg"

MORPHI_ROOT="${CELL_SEG_ROOT}/Morphi"

DOANET_ROOT="${MORPHI_ROOT}/DoANet"

INSTANCE_ROOT="${MORPHI_ROOT}/Instance-segment"

PIPELINE_ROOT="${CELL_SEG_ROOT}/pipeline"


# ============================================================
# Python executables
# ============================================================

DOANET_PYTHON="python"

INSTANCE_PYTHON="python"


# ============================================================
# Scripts
# ============================================================

DOANET_SCRIPT="${DOANET_ROOT}/inference_single_image.py"

CONVERT_SCRIPT="${PIPELINE_ROOT}/convert_semantic_mask.py"

INSTANCE_SCRIPT="${INSTANCE_ROOT}/tools/inference/inference_with_LIVE_MODEL.py"


# ============================================================
# Input
# ============================================================

IMAGE_PATH="/storage/jinxinLab/wangyimin/Cell_seg/Datasets/test/test_1080.png"


# ============================================================
# DoANet model
# ============================================================

DOANET_CHECKPOINT="${DOANET_ROOT}/output/inference.pth"

DOANET_CFG="${DOANET_ROOT}/configs/your_config.yaml"

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

INSTANCE_IMAGE_DIR="${INSTANCE_ROOT_A}/${CASE_NAME}/channel/more_bf"

INSTANCE_SEMANTIC_DIR="${INSTANCE_ROOT_B}/${CASE_NAME}/combined"

INSTANCE_RESULT_DIR="${INSTANCE_ROOT_A}/${CASE_NAME}/channel/output_r02c02"


# ============================================================
# Converted semantic mask
# ============================================================

SEMANTIC_MASK="${INSTANCE_SEMANTIC_DIR}/${IMAGE_STEM}.png"


# ============================================================
# Final result
# ============================================================

FINAL_CASE_DIR="${FINAL_OUTPUT}/${CASE_NAME}"