#!/bin/bash
EPOCH_TIME=${epoch_time:-235}
OUT_DIR=${out_dir:-'./model_out20260211_padding'}
CFG=${cfg:-'configs/swin_tiny_patch4_window7_224_lite.yaml'}
ROOT_DIR=${root_dir:-'/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928'}
LEARNING_RATE=${learning_rate:-0.001}
IMG_SIZE=${img_size:-224}
BATCH_SIZE=${batch_size:-24}

echo "start batch inference"
echo "ROOT_DIR: $ROOT_DIR"
echo "OUT_DIR: $OUT_DIR"

# LIST_DIR="$DATA_DIR/patches.txt"
# find "$DATA_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tif" -o -name "*.tiff" -o -name "*.jepg" \) | sort > "$LIST_DIR"

find "$ROOT_DIR" -type d -path "*/BF_r02c22/224_patches_all/brightfield" | while read -r DATA_DIR; do
    echo "======================================"
    echo "Processing: $DATA_DIR"

    LIST_DIR="$DATA_DIR/patches.txt"
    echo "List dir is: $LIST_DIR"

    find "$DATA_DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.tif" -o -name "*.tiff" -o -name "*.jpeg" \) | sort > "$LIST_DIR"


    python inference_softmax.py \
      --dataset Organoids \
      --cfg "$CFG" \
      --is_saveni \
      --volume_path "$DATA_DIR" \
      --max_epochs "$EPOCH_TIME" \
      --output_dir "$OUT_DIR" \
      --img_size "$IMG_SIZE" \
      --base_lr "$LEARNING_RATE" \
      --batch_size "$BATCH_SIZE" \
      --list_dir "$LIST_DIR"
done

echo "All done."