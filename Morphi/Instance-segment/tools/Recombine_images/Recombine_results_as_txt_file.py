import os
import cv2
import numpy as np
from collections import defaultdict
import shutil


def reconstruct_one_case(coord_file, patch_folder, output_folder, image_size=(1080, 1080)):
    """
    according to patch_coords.txt and patch_folder's patch result，to reconstruct the image。
    """
    if not os.path.exists(coord_file):
        print(f"Skip: coord file not found: {coord_file}")
        return

    if not os.path.isdir(patch_folder):
        print(f"Skip: patch folder not found: {patch_folder}")
        return

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    patch_dict = defaultdict(list)

    with open(coord_file, "r") as f:
        header_checked = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            # skip table head
            if not header_checked:
                header_checked = True
                if line.lower().startswith("filename,"):
                    continue

            filename, x1, y1, x2, y2 = line.split(",")
            filename = os.path.splitext(filename)[0] + "_pred_unpad.png"

            base_name = filename.split("_patch_")[0]
            patch_dict[base_name].append((filename, int(x1), int(y1), int(x2), int(y2)))

    for base_name, patch_list in patch_dict.items():
        canvas = np.zeros(image_size, dtype=np.uint8)

        for fname, x1, y1, x2, y2 in patch_list:
            patch_path = os.path.join(patch_folder, fname)
            patch = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)

            if patch is None:
                print(f"Warning: cannot read {patch_path}")
                continue

            patch = patch.astype(np.uint8)

            h = y2 - y1
            w = x2 - x1

            if patch.shape != (h, w):
                patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_NEAREST)

            target = canvas[y1:y2, x1:x2]
            canvas[y1:y2, x1:x2] = np.maximum(target, patch)

        out_path = os.path.join(output_folder, f"{base_name}_mask_reconstructed.png")
        cv2.imwrite(out_path, canvas)
        print(f"Saved reconstructed mask: {out_path}")


def find_case_name_from_coord_path(coord_file, root_a):
    """
    从
    xxxxxxa/Group.../channel/ch1/224_patches_all/brightfield/patch_coords.txt
    extract Group... file name
    """
    rel_path = os.path.relpath(coord_file, root_a)
    parts = rel_path.split(os.sep)

    if len(parts) < 2:
        return None

    return parts[0]



def batch_reconstruct(root_a, root_b, image_size=(2000, 2000)):
    """
    Batch process:
    - Find all patch_coords.txt in root_a
    - Extract the case_name
    - Match the case folder with the same name in root_b
    - Read root_b/case_name/prob
    - Output to root_b/case_name/combined
    """
    coord_files = []

    for dirpath, dirnames, filenames in os.walk(root_a):
        if "patch_coords.txt" in filenames:
            full_path = os.path.join(dirpath, "patch_coords.txt")
            # Only keep patch_coords.txt in bf
            if os.path.normpath(full_path).endswith(
                os.path.normpath("BF_r02c22/224_patches_all/brightfield/patch_coords.txt")
            ):
                coord_files.append(full_path)

    coord_files = sorted(coord_files)

    print(f"Found {len(coord_files)} coord files.")

    for coord_file in coord_files:
        case_name = find_case_name_from_coord_path(coord_file, root_a)
        if case_name is None:
            print(f"Skip: cannot parse case name from {coord_file}")
            continue

        patch_folder = os.path.join(root_b, case_name, "pred")
        output_folder = os.path.join(root_b, case_name, "combined")

        # Check if the 'combined' folder already exists. If so, skip this case.
        if os.path.exists(output_folder):
            print(f"Skip: 'combined' already exists for case {case_name}. Skipping.")
            continue

        print("\n" + "=" * 80)
        print(f"Case: {case_name}")
        print(f"coord_file:   {coord_file}")
        print(f"patch_folder: {patch_folder}")
        print(f"output_folder:{output_folder}")

        reconstruct_one_case(
            coord_file=coord_file,
            patch_folder=patch_folder,
            output_folder=output_folder,
            image_size=image_size
        )

    print("\nAll done.")


if __name__ == "__main__":
    root_a = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928"
    root_b = "/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/model_out20260211_padding/20260712"

    batch_reconstruct(
        root_a=root_a,
        root_b=root_b,
        image_size=(1080,1080)
    )