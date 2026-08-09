import os
import re
from typing import Optional
import cv2
import numpy as np

# Path
img_dir = '/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/Drugscreen_datasets/Datasets_20250421_96_treat/ch2'
mask_dir = '/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/Drugscreen_datasets/Datasets_20250421_96_treat/ch2/mask'

out_crop_img_dir = '/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/Drugscreen_datasets/Datasets_20250421_96_treat/patches/img'
out_crop_mask_dir = '/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/Drugscreen_datasets/Datasets_20250421_96_treat/patches/mask'
out_positions_path = '/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/Drugscreen_datasets/Datasets_20250421_96_treat/patches/positions.txt'

os.makedirs(out_crop_img_dir, exist_ok=True)
os.makedirs(out_crop_mask_dir, exist_ok=True)

# Param
min_area = 100      # ignore small area
bbox_margin = 0    # Extra margins (pixels) for bounding boxes

IMG_EXTS = ['.tiff', '.tif', '.png', '.jpg', '.jpeg']
mask_pattern = re.compile(r'^(?P<base>r\d{2}c\d{2}f\d{2})_gtmask\.png$', re.IGNORECASE)

def find_image_for_base(base_name: str) -> Optional[str]:
    for ext in ['.tiff', '.tif']:
        cand = os.path.join(img_dir, base_name + ext)
        if os.path.exists(cand):
            return cand
    for ext in IMG_EXTS:
        cand = os.path.join(img_dir, base_name + ext)
        if os.path.exists(cand):
            return cand
    return None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# initial
if not os.path.exists(out_positions_path):
    with open(out_positions_path, 'w', encoding='utf-8') as f:
        f.write('base,idx,x0,y0,x1,y1,width,height,area,img_out,mask_out\n')

# main
for mask_name in os.listdir(mask_dir):
    m = mask_pattern.match(mask_name)
    if not m:
        continue

    base = m.group('base')
    mask_path = os.path.join(mask_dir, mask_name)

    img_path = find_image_for_base(base)
    if img_path is None:
        print(f"Skip: can not find original image: {base}.*")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Skip: failed to read image: {img_path}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Skip: failed to read mask: {mask_path}")
        continue

    if (img.shape[0] != mask.shape[0]) or (img.shape[1] != mask.shape[1]):
        print(f"Skip: sizes are not equal: {img_path} vs {mask_path}")
        continue

    H, W = mask.shape[:2]
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    kept = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        x0 = clamp(x - bbox_margin, 0, W - 1)
        y0 = clamp(y - bbox_margin, 0, H - 1)
        x1 = clamp(x + w + bbox_margin, 0, W)
        y1 = clamp(y + h + bbox_margin, 0, H)

        crop_img = img[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]

        kept += 1
        img_out_name = f"{base}_crop{i}.png"
        mask_out_name = f"{base}_crop{i}.png"

        img_out_path = os.path.join(out_crop_img_dir, img_out_name)
        mask_out_path = os.path.join(out_crop_mask_dir, mask_out_name)

        cv2.imwrite(img_out_path, crop_img)
        cv2.imwrite(mask_out_path, crop_mask)

        # record position
        with open(out_positions_path, 'a', encoding='utf-8') as f:
            f.write(f"{base},{i},{x0},{y0},{x1},{y1},{x1-x0},{y1-y0},{area},{img_out_name},{mask_out_name}\n")

    print(f"Finished {base}: components {num_labels-1}, saved {kept} patches; location already recorded.")
