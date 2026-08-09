import os
import sys
import cv2
import torch
import tifffile as tiff
import numpy as np
import pandas as pd


def process_one_case(input_folder, output_folder, semantic_mask_folder):
    print(f"\n=== Processing case ===")
    print(f"input_folder: {input_folder}")
    print(f"output_folder: {output_folder}")
    print(f"mask_folder: {semantic_mask_folder}")

    os.makedirs(output_folder, exist_ok=True)

    overlay_folder = os.path.join(output_folder, "semantic_overlay")
    os.makedirs(overlay_folder, exist_ok=True)

    xlsx_path = os.path.join(output_folder, "semantic_summary.xlsx")
    if os.path.exists(xlsx_path):
        os.remove(xlsx_path)

    results = []
    valid_exts = (".jpg", ".png", ".tif", ".jpeg", ".tiff")

    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(valid_exts) or filename.startswith("."):
            continue

        input_path = os.path.join(input_folder, filename)

        image = cv2.imread(input_path)
        if image is None:
            try:
                image = tiff.imread(input_path)
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis=-1)
                if image.dtype != np.uint8:
                    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Cannot read image: {input_path}, error: {e}")
                continue

        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(overlay_folder, filename)

        semantic_mask_path = find_mask_path(semantic_mask_folder, stem)
        semantic_mask = read_mask(semantic_mask_path)

        if semantic_mask is None:
            print(f"No semantic mask found for {filename}")
            continue

        if semantic_mask.ndim == 3:
            semantic_mask = semantic_mask[:, :, 0]

        if semantic_mask.shape[:2] != image.shape[:2]:
            semantic_mask = cv2.resize(
                semantic_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # semantic classes
        # 127 = live
        # 254 = dead
        live_mask = semantic_mask == 127
        dead_mask = semantic_mask == 254

        live_pixels = int(np.sum(live_mask))
        dead_pixels = int(np.sum(dead_mask))
        total_pixels = live_pixels + dead_pixels
        death_rate = dead_pixels / total_pixels if total_pixels > 0 else np.nan

        vis_image = image.copy()
        alpha = 0.35

        # live: green
        vis_image[live_mask] = (
            vis_image[live_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha
        ).astype(np.uint8)

        # dead: red
        vis_image[dead_mask] = (
            vis_image[dead_mask] * (1 - alpha) + np.array([0, 0, 255]) * alpha
        ).astype(np.uint8)

        # optional image-level text
        if np.isnan(death_rate):
            text = "DR: NA"
        else:
            text = f"DR: {death_rate:.3f}"

        cv2.rectangle(vis_image, (5, 5), (150, 32), (0, 0, 0), -1)
        cv2.putText(
            vis_image,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imwrite(output_path, vis_image)
        print(f"Saved semantic overlay: {output_path}")

        results.append({
            "image_name": filename,
            "live_pixels": live_pixels,
            "dead_pixels": dead_pixels,
            "total_semantic_pixels": total_pixels,
            "death_rate": death_rate
        })

    df = pd.DataFrame(results)
    df.to_excel(xlsx_path, index=False)
    print(f"Saved semantic summary: {xlsx_path}")
    
def find_mask_path(mask_folder, filename_stem):
    """
    Automatic match by filename live/dead mask
    support:
      xxx.png
      xxx.tif
      xxx.tiff
      xxx_mask.png
      xxx_mask.tif
      xxx_mask.tiff
    """
    candidates = [
        os.path.join(mask_folder, filename_stem + ".png"),
        os.path.join(mask_folder, filename_stem + ".tif"),
        os.path.join(mask_folder, filename_stem + ".tiff"),
        os.path.join(mask_folder, filename_stem + "_mask.png"),
        os.path.join(mask_folder, filename_stem + "_mask.tif"),
        os.path.join(mask_folder, filename_stem + "_mask.tiff"),
        os.path.join(mask_folder, filename_stem + "_mask_reconstructed.png")
    ]
    # print(candidates)

    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def read_mask(mask_path):
    """
    read live/dead mask
    require mask:
      0 = background
      1 = live
      2 = dead
    """
    if mask_path is None:
        return None

    if mask_path.lower().endswith((".tif", ".tiff")):
        mask = tiff.imread(mask_path)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    if mask is None:
        return None

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask.astype(np.uint8)




def find_case_folders_in_root_a(root_a):
    """
    find all root_a/<case_name>/channel/ch1
    :return [(case_name, input_folder, output_folder), ...]
    """
    cases = []
    for name in sorted(os.listdir(root_a)):
        case_dir = os.path.join(root_a, name)
        if not os.path.isdir(case_dir):
            continue

        input_folder = os.path.join(case_dir, "channel", "BF")
        output_folder = os.path.join(case_dir, "channel", "output")

        if os.path.isdir(input_folder):
            cases.append((name, input_folder, output_folder))
    return cases


def batch_process(root_a, root_b):
    cases = find_case_folders_in_root_a(root_a)
    print(f"Found {len(cases)} candidate cases in root_a")

    matched = 0
    for case_name, input_folder, output_folder in cases:
        # Skip processing if the output folder already exists
        if os.path.isdir(output_folder):
            print(f"Skip {case_name}: output folder already exists -> {output_folder}")
            continue

        semantic_mask_folder = os.path.join(root_b, case_name, "combined")

        if not os.path.isdir(semantic_mask_folder):
            print(f"Skip {case_name}: mask folder not found -> {semantic_mask_folder}")
            continue

        matched += 1
        process_one_case(
            input_folder=input_folder,
            output_folder=output_folder,
            semantic_mask_folder=semantic_mask_folder,
            # demo=demo
        )

    print(f"\nMatched and processed {matched} cases.")
    
    

def main():
    root_a_default = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928"
    root_b_default = "/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/model_out20260211_padding/20260507"

    root_a = sys.argv[1] if len(sys.argv) > 1 else root_a_default
    root_b = sys.argv[2] if len(sys.argv) > 2 else root_b_default

    print(f"root_a: {root_a}")
    print(f"root_b: {root_b}")
    
    batch_process(root_a, root_b)
    
    
    
if __name__ == "__main__":
    main()