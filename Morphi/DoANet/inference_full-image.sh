#!/bin/bash

set -e

# Paths
PYTHON_SCRIPT="inference_single_image.py"
IMAGE_PATH="/storage/jinxinLab/wangyimin/Cell_seg/Datasets/your_image.png"
OUTPUT_DIR="/storage/jinxinLab/wangyimin/Cell_seg/your_model_output"
CHECKPOINT="${OUTPUT_DIR}/inference.pth"
TEST_SAVE_DIR="/storage/jinxinLab/wangyimin/Cell_seg/predictions/single_image"
CFG="/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/configs/your_config.yaml"

# Parameters
IMG_SIZE=224
STRIDE=224
NUM_CLASSES=3

# GPU
export CUDA_VISIBLE_DEVICES=0

# Run
python ${PYTHON_SCRIPT} \
    --image_path "${IMAGE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --test_save_dir "${TEST_SAVE_DIR}" \
    --cfg "${CFG}" \
    --img_size ${IMG_SIZE} \
    --stride ${STRIDE} \
    --num_classes ${NUM_CLASSES}