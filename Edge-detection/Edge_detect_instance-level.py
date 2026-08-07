import os
import glob
import time

import cv2
import numpy as np
from cluster_dbscan import cluster_edges_with_dbscan
from instance_mask import make_instance_mask_from_cluster_label_map


# =========================
# Input/output/label path
# =========================
image_folder = "/Volumes/Expansion/Flowchart_materials/Figure2_AO_PI_analysis_flowchart/basic_benchmark/Alive_case/20241205_72h/samples/r15c12f04"
output_path = "/Volumes/Expansion/Flowchart_materials/Figure2_AO_PI_analysis_flowchart/basic_benchmark/Alive_case/20241205_72h/samples/r15c12f04/AO_output"
label_path = "/Volumes/Expansion/Flowchart_materials/Figure2_AO_PI_analysis_flowchart/basic_benchmark/Alive_case/20241205_72h/samples/r15c12f04/AO_label"


# =========================
"""
param
"""

# canny detection
canny_low = 60
canny_high = 80

# DBSCAN cluster
dbscan_eps = 2.5
dbscan_min_samples = 1
dbscan_min_cluster_size = 50

# area threshold
min_area = 50
max_area = 1000000

# True: final instance id will keep DBSCAN cluster id
# False: final instance id renumber to 1, 2, 3...
keep_original_cluster_id = False


# =========================
# tools
# =========================
def get_image_paths(image_folder):
    image_paths = []
    image_paths.extend(glob.glob(os.path.join(image_folder, "*.tif")))
    image_paths.extend(glob.glob(os.path.join(image_folder, "*.tiff")))
    image_paths = sorted(image_paths)
    return image_paths

def colorize_instance_mask(instance_mask, seed=42):
    """
    Convert instance label mask to a colored RGB/BGR visualization.

    instance_mask:
        0 = background
        1, 2, 3... = instance labels

    return:
        color_mask_bgr: uint8 BGR image for cv2.imwrite
    """
    rng = np.random.default_rng(seed)

    h, w = instance_mask.shape

    color_mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    labels = np.unique(instance_mask)
    labels = labels[labels > 0]

    for label_id in labels:
        color = rng.integers(
            low=50,
            high=256,
            size=3,
            dtype=np.uint8
        )

        color_mask_rgb[instance_mask == label_id] = color

    color_mask_bgr = cv2.cvtColor(
        color_mask_rgb,
        cv2.COLOR_RGB2BGR
    )

    return color_mask_bgr


def read_image_gray_and_display(image_path):
    """
    :return：
        gray: uint8 grayscale image, for Canny
        display_image: BGR image, for overlay
        raw_image: cv2 for read original image
    """
    raw_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if raw_image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if raw_image.ndim == 2:
        gray_float = raw_image.astype(np.float32)

        gray_u8 = normalize_to_uint8(
            gray_float,
            low=1,
            high=99.5
        )

        display_image = cv2.cvtColor(
            gray_u8,
            cv2.COLOR_GRAY2BGR
        )

    else:
        # cv2 import as BGR/BGRA
        if raw_image.shape[2] == 4:
            bgr = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = raw_image.copy()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_u8 = normalize_to_uint8(
            gray.astype(np.float32),
            low=1,
            high=99.5
        )

        # display compress to uint8，avoid 16-bit displaying an exception
        display_image = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    return gray_u8, display_image, raw_image


def normalize_to_uint8(img, low=1, high=99.5):
    img = img.astype(np.float32)

    vmin, vmax = np.percentile(img, [low, high])

    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)

    img_n = (img - vmin) / (vmax - vmin)
    img_n = np.clip(img_n, 0, 1)

    return (img_n * 255).astype(np.uint8)


def make_instance_vis(instance_mask):
    """
    Showing instance mask。
    notice: only for check
    """

    if np.max(instance_mask) == 0:
        return np.zeros_like(instance_mask, dtype=np.uint8)

    instance_vis = cv2.normalize(
        instance_mask,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return instance_vis


# =========================
# Main
# =========================
def main():
    # start_time = time.time()

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    image_paths = get_image_paths(image_folder)

    print(f"Found images: {len(image_paths)}")

    for i, image_path in enumerate(image_paths):

        image_start = time.time()

        basename = os.path.basename(image_path)
        name_no_ext = os.path.splitext(basename)[0]

        print(f"\nProcessing: {basename}")

        # =========================
        # Read image
        # =========================
        gray_u8, display_image, raw_image = read_image_gray_and_display(
            image_path
        )

        cv2.imwrite(
            os.path.join(output_path, f"{name_no_ext}_input_gray.tiff"),
            gray_u8
        )

        # =========================
        # Canny edge detection
        # =========================
        edges = cv2.Canny(
            gray_u8,
            canny_low,
            canny_high
        )

        cv2.imwrite(
            os.path.join(output_path, f"{name_no_ext}_canny_edges.tiff"),
            edges
        )

        # =========================
        # DBSCAN
        # =========================
        clustered_img, cluster_label_map, cluster_info = cluster_edges_with_dbscan(
            output_path=output_path,
            edges=edges,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            min_cluster_size=dbscan_min_cluster_size,
            save_label_map=True,
            prefix=name_no_ext
        )

        cv2.imwrite(
            os.path.join(output_path, f"{name_no_ext}_cluster_image.tiff"),
            cv2.cvtColor(clustered_img, cv2.COLOR_RGB2BGR)
        )

        # =========================
        # Using original cluster_label_map to generate instance mask
        # =========================
        instance_mask, binary_mask, overlay = make_instance_mask_from_cluster_label_map(
            cluster_label_map=cluster_label_map,
            display_image=display_image,
            min_area=min_area,
            max_area=max_area,
            keep_original_cluster_id=keep_original_cluster_id
        )

        # =========================
        # Saving result
        # =========================
        instance_file_path = os.path.join(
            label_path,
            f"{name_no_ext}_instance_mask.tiff"
        )

        binary_file_path = os.path.join(
            label_path,
            f"{name_no_ext}_binary_mask.png"
        )

        instance_vis_file_path = os.path.join(
            label_path,
            f"{name_no_ext}_instance_mask_vis.png"
        )

        overlay_file_path = os.path.join(
            output_path,
            f"{name_no_ext}_overlay.tiff"
        )

        instance_color_file_path = os.path.join(
            label_path,
            f"{name_no_ext}_instance_mask_color.png"
        )

        instance_color = colorize_instance_mask(
            instance_mask,
            seed=42
        )

        cv2.imwrite(
            instance_color_file_path,
            instance_color
        )

        # True instance mask：for analysis / benchmark
        cv2.imwrite(
            instance_file_path,
            instance_mask.astype(np.uint16)
        )

        # showing instance mask
        instance_vis = make_instance_vis(instance_mask)

        cv2.imwrite(
            instance_vis_file_path,
            instance_vis
        )

        # binary mask
        cv2.imwrite(
            binary_file_path,
            binary_mask
        )

        # overlay
        cv2.imwrite(
            overlay_file_path,
            overlay
        )

        num_instances = len(np.unique(instance_mask)) - 1

        image_end = time.time()

        print(f"Detected instances: {num_instances}")
        print(f"Saved instance mask: {instance_file_path}")
        print(f"Saved instance vis: {instance_vis_file_path}")
        print(f"Saved binary mask: {binary_file_path}")
        print(f"Saved overlay: {overlay_file_path}")
        print(f"Time: {image_end - image_start:.2f} s")

    # end_time = time.time()
    # print(f"\nTotal inference time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()