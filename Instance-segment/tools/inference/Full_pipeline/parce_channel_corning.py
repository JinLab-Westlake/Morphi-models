import os, re, cv2, imageio
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from skimage import measure
from matplotlib import pyplot as plt
from inference_pipe import main



"""
Filename matching rule

Example:
AssayPlate_Corning_corning_3831_A06.b1_T0001F001L01A02Z02C02.tif

region  = A06.b1
layer   = 2       (Z01-Z06)
channel = 2       (C01/C1 or C02/C2)
"""
filename_pattern = re.compile(
    r"(?P<region>[A-Z]\d{2}\.[A-Z]\d+)"
    r"_T(?P<time>\d+)"
    r"F(?P<field>\d+)"
    r"L(?P<location>\d+)"
    r"A(?P<area>\d+)"
    r"Z0?(?P<layer>[1-6])"
    r"C0?(?P<channel>[12])"
    r"(?=\.tiff?$)",
    re.IGNORECASE,
)


def parse_filename(filename):
    """Parse region, layer and channel from one image filename."""
    match = filename_pattern.search(filename)
    if match is None:
        return None

    return {
        "region": match.group("region"),
        "layer": int(match.group("layer")),
        "channel": f"ch{int(match.group('channel'))}",
    }


"""
Sobel operation
"""
def calculate_sobel_sharpness(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobel_x, sobel_y)
    return sobel



"""
cluster samples
"""
def cluster_edges_with_dbscan(edges, eps=5, min_samples=5,min_cluster_size=50):
    
    if edges is None:
        print("Warning: edges is None, skip this image.")
        return None, None  # 或者 return 空数组
    
    
    # 获取边缘点坐标
    y_coords, x_coords = np.where(edges > 0)
    edge_points = np.column_stack((x_coords, y_coords))

    # 如果没有边缘点，直接返回一张空图像
    if edge_points.shape[0] == 0:
        print(" No edges found in the image")
        return None

    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(edge_points)
    index = dbscan.labels_

    # 创建彩色聚类结果图像
    clustered_img = np.zeros((*edges.shape, 3), dtype=np.uint8)
    clustered_img_sample = np.zeros((*edges.shape, 3), dtype=np.uint8)
    unique_labels = set(labels)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    count = 0
    for label in unique_labels:
        if label == -1:
            continue  # 忽略噪声点
        # 计算当前聚类的点数
        points_in_cluster = np.sum(labels == label)
        if points_in_cluster < min_cluster_size:
            continue
        color = np.array(colors(label)[:3]) * 255  # 选择一个颜色并转换为RGB
        clustered_img[edge_points[labels == label, 1], edge_points[labels == label, 0]] = color

        clustered_img_sample[edge_points[labels == label, 1], edge_points[labels == label, 0]] = color

        # 计算聚类中心
        clustered_center = np.mean(edge_points[labels ==label],axis=0).astype(int)

        # 在聚类中心标注index
        cv2.putText(clustered_img_sample,str(label),(clustered_center[0], clustered_center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        count += 1


    print(f"Number of clusters:{count}")
    return clustered_img



def process_channel_images(image, image_name, output_file_path, eps=3, min_samples=5, min_cluster_size=10):
    # """Process images for a given channel"""
    # # Load images from the channel folder
    # images, image_paths = load_images(image_folder)

    output_path = output_file_path/"SYTOX_output"
    label_path = output_file_path/"SYTOX_labels"
    
    # Ensure output and label folders exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    # Adjust image (e.g., adjust contrast and brightness)
    alpha = 2  # Contrast parameter
    beta = 1  # Brightness parameter
    
    # print(image.shape)
    if len(image.shape) == 2:  # Grayscale image (1 channel)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    equalized_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Edge detection
    edges = cv2.Canny(equalized_image, 70, 130)
    
    # Cluster edges using DBSCAN
    clustered_edges = cluster_edges_with_dbscan(edges, eps=eps, min_samples=min_samples, min_cluster_size=min_cluster_size)

    if clustered_edges is not None:
        # print(f"Skipping {os.path.basename(image_path)} due to no edges found.")
        # continue

        # Dilate the clustered edges
        kernel = np.ones((3, 3), np.uint8)
        clustered_edges = cv2.dilate(clustered_edges, kernel, iterations=1)

        # Find connected components in the clustered image
        labels = measure.label(clustered_edges[..., 0], connectivity=2)
        regions = measure.regionprops(labels, intensity_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # Create a blank label image
        height, width = image.shape[:2]
        label_image = np.zeros((height, width), dtype=np.uint8)

        # Process each region (contour) in the image
        for region in regions:
            if len(region.bbox) == 4:  # Ensure bbox length is 4
                minr, minc, maxr, maxc = region.bbox
                region_mask = np.zeros_like(image[..., 0], dtype=np.uint8)
                region_mask[minr:maxr, minc:maxc] = region.filled_image

                # Draw contours
                contours, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

                # Draw contours on the label image
                for contour in contours:
                    cv2.drawContours(label_image, [contour], -1, 2, thickness=cv2.FILLED)

        # Save the label image
        label_file_path = f"{label_path}/{image_name}.tiff"
        cv2.imwrite(label_file_path, label_image * 255)

        # Save the processed image with clusters
        output_file_path = f"{output_path}/{image_name}.tiff"
        cv2.imwrite(output_file_path, image)

        print(f"Finished processing {image_name}")
        
        return label_image




def process_image_group(
    channel_key,
    region_key,
    matching_files,
    image_file_path,
    output_file_path,
    expected_layers=6,
):
    try:
        if len(matching_files) != expected_layers:
            print(
                f"Warning: {region_key} {channel_key} contains "
                f"{len(matching_files)} layers instead of {expected_layers}; skip."
            )
            return None

        sharpness_stack = []
        image_stack = []

        for file in matching_files:
            image_path = os.path.join(image_file_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")

            image_stack.append(image)
            sharpness_stack.append(
                calculate_sobel_sharpness(image)
            )

        sharpness_stack = np.asarray(sharpness_stack)
        image_stack = np.asarray(image_stack)

        # Select, pixel by pixel, the intensity from the Z layer
        # with the largest Sobel response.
        sharpest_indices = np.argmax(
            sharpness_stack,
            axis=0,
        )

        rows, cols = np.indices(sharpest_indices.shape)
        composite_image = image_stack[
            sharpest_indices,
            rows,
            cols,
        ]

        channel_folder = os.path.join(
            output_file_path,
            channel_key,
        )
        os.makedirs(channel_folder, exist_ok=True)

        composite_path = os.path.join(
            channel_folder,
            f"{region_key}.tiff",
        )
        cv2.imwrite(composite_path, composite_image)
        print(f"Processed {channel_key} - {region_key}")

        # Existing downstream definition:
        # ch2 = SYTOX; ch1 = bright-field.
        if channel_key == "ch2":
            label_image = process_channel_images(
                composite_image,
                region_key,
                output_file_path,
            )
            print(f"Processed SYTOX image: {channel_key}")
            return label_image

        if channel_key == "ch1":
            print(f"Processed bright-field image: {channel_key}")
            return composite_image

        raise ValueError(f"Unsupported channel: {channel_key}")

    except Exception as e:
        print(
            f"An error occurred while processing "
            f"{channel_key} - {region_key}: {e}"
        )
        raise


def split_images_by_channel(
    input_dir,
    output_root_name="channel",
):
    input_dir = Path(input_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_dir}"
        )
    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a folder: {input_dir}"
        )

    output_root = input_dir.parent / output_root_name
    output_root.mkdir(parents=True, exist_ok=True)

    # {(region, channel): {layer_number: filename}}
    matching_files_dict = {}

    for file in os.listdir(input_dir):
        file_path = input_dir / file

        if not file_path.is_file() or file.startswith("."):
            continue
        if file_path.suffix.lower() not in {
            ".tif",
            ".tiff",
        }:
            continue

        parsed = parse_filename(file)
        if parsed is None:
            print(
                "Filename does not match the expected rule, "
                f"skip: {file}"
            )
            continue

        region_key = parsed["region"]
        channel_key = parsed["channel"]
        layer_index = parsed["layer"]
        key = (region_key, channel_key)

        matching_files_dict.setdefault(key, {})

        if layer_index in matching_files_dict[key]:
            previous_file = matching_files_dict[key][
                layer_index
            ]
            raise ValueError(
                f"Duplicate layer for {region_key} "
                f"{channel_key} Z{layer_index:02d}: "
                f"{previous_file} and {file}"
            )

        matching_files_dict[key][layer_index] = file

    # Process each Axx.bx region only once.
    region_keys = sorted({
        region
        for region, _ in matching_files_dict
    })

    for region_key in region_keys:
        ch1_by_layer = matching_files_dict.get(
            (region_key, "ch1"),
            {},
        )
        ch2_by_layer = matching_files_dict.get(
            (region_key, "ch2"),
            {},
        )

        # Explicit order: Z01, Z02, ..., Z06.
        ch1_files = [
            ch1_by_layer[layer]
            for layer in sorted(ch1_by_layer)
        ]
        ch2_files = [
            ch2_by_layer[layer]
            for layer in sorted(ch2_by_layer)
        ]

        print(
            f"{region_key}: "
            f"ch1 layers={sorted(ch1_by_layer)}, "
            f"ch2 layers={sorted(ch2_by_layer)}"
        )

        # C02 -> ch2 -> SYTOX
        sytox_label = process_image_group(
            "ch2",
            region_key,
            ch2_files,
            input_dir,
            output_root,
        )

        # C01/C1 -> ch1 -> bright-field
        bf_image = process_image_group(
            "ch1",
            region_key,
            ch1_files,
            input_dir,
            output_root,
        )

        # if (
        #     sytox_label is not None
        #     and bf_image is not None
        # ):
        #     main(
        #         region_key,
        #         sytox_label,
        #         bf_image,
        #         output_root,
        #     )


# Example usage
if __name__ == "__main__":
    input_dir = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250714_PE/20250713-YOKOWGAWA-A2780-24drugs-1536-test_20250713_121140/AssayPlate_Corning_corning_3831"
    split_images_by_channel(input_dir)