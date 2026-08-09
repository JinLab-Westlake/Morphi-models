import os
import cv2
import numpy as np
from collections import defaultdict
import shutil

# input path
coord_file = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928/20240923_A2780_pre_drug_onepeak__2024-09-23T19_08_40-Measurement/channel/BF/224_patches_all/brightfield/patch_coords.txt"
patch_folder = "/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/model_out20260211_padding/20260507/20240923_A2780_pre_drug_onepeak__2024-09-23T19_08_40-Measurement/pred" 
output_folder = "/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/model_out20260211_padding/20260507/20240923_A2780_pre_drug_onepeak__2024-09-23T19_08_40-Measurement/recombined"
# os.makedirs(output_folder, exist_ok=True)


if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)


# output size
image_size = (1080, 1080)
# image_size = (1980,1980)

# read patch location
patch_dict = defaultdict(list)

with open(coord_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        filename, x1, y1, x2, y2 = line.split(",")
        base = filename.rsplit(".",1)[0]
        filename = base + "_pred_unpad.png"
        # filename = filename + "_pred.png"
        # base_name = filename.split("_mask_patch_")[0]
        base_name = filename.split('_patch_')[0]
        # base_name = filename.split('_inst')[0]
        
        patch_dict[base_name].append((filename, int(x1), int(y1), int(x2), int(y2)))

# Number and splice each original image
for base_name, patch_list in patch_dict.items():
    canvas = np.zeros(image_size, dtype=np.uint8)

    for fname, x1, y1, x2, y2 in patch_list:
        patch_path = os.path.join(patch_folder, fname)
        patch = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
        if patch is None:
            print(f"Warning: cannot read {patch_path}")
            continue

        # patch = (patch > 0).astype(np.uint8) * 255  # 二值化（保证0或255）
        # ph, pw = patch.shape
        
        ph, pw = patch.shape

        # Uniform coercion to np.uint8 to prevent abnormal floating-point values
        patch = patch.astype(np.uint8)

        # Target region extraction
        target = canvas[y1:y2, x1:x2]

        # Make sure the dimensions match
        if patch.shape != (y2 - y1, x2 - x1):
            patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        # Merge multiple classes
        canvas[y1:y2, x1:x2] = np.maximum(target, patch)
    
        # Safely crop the matching size
        target = canvas[y1:y2, x1:x2]
        patch_resized = cv2.resize(patch, (x2 - x1, y2 - y1))
        canvas[y1:y2, x1:x2] = np.maximum(target, patch_resized)

    out_path = os.path.join(output_folder, f"{base_name}_mask_reconstructed.png")
    cv2.imwrite(out_path, canvas)
    print(f"Saved reconstructed mask: {out_path}")
    
    