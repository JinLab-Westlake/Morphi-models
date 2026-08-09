import os, re, cv2, imageio
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from skimage import measure
from matplotlib import pyplot as plt
from inference_pipe import main



"""
match rule
"""
# Define a pattern for 'r\d{2}c\d{2}f\d{2}' (e.g., r01c01f01)
pattern = re.compile(r"r\d{2}c\d{2}f\d{2}")

# Define pattern for channel information (e.g., ch1, channel1)
# channel_pattern = re.compile(r"(ch|channel)\d+[a-zA-Z]*", re.IGNORECASE)
channel_pattern = re.compile(r"(ch\d+)", re.IGNORECASE)


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
        output_file_path = f"{output_path}/ {image_name}.tiff"
        cv2.imwrite(output_file_path, image)

        print(f"Finished processing {image_name}")
        
        return label_image




def process_image_group(channel_key, region_key, matching_files, image_file_path, output_file_path):
    try:
        if len(matching_files) < 1:
            print(f"Not enough images for {channel_key}")
            return

        # sorted by name
        matching_files.sort()
        sharpness_stack = []

        image_stack = [cv2.imread(os.path.join(image_file_path, file), cv2.IMREAD_GRAYSCALE) for file in matching_files]
        for image in image_stack:
            sharpness = calculate_sobel_sharpness(image)
            sharpness_stack.append(sharpness)

        # transfer to array
        sharpness_stack = np.array(sharpness_stack)
        image_stack = np.array(image_stack)

        # 获取每个像素最清晰的层的索引
        sharpest_indices = np.argmax(sharpness_stack, axis=0)

        # 创建合成图像
        height, width = sharpest_indices.shape
        composite_image = np.zeros_like(image_stack[0])

        for i in range(height):
            for j in range(width):
                composite_image[i, j] = image_stack[sharpest_indices[i, j], i, j]
        
        channel_folder = os.path.join(output_file_path, channel_key) # key0 is the channel number
        os.makedirs(channel_folder, exist_ok = True)
        
        # save the processed composite image
        cv2.imwrite(f"{output_file_path}/{channel_key}/{region_key}.tiff", composite_image)
        print(f"Processed {channel_key} - {region_key}")
        
        # add the logic to process channel_1 sytox images
        if channel_key == 'ch2':
            label_image = process_channel_images(composite_image,region_key,output_file_path)
            print(f"have processed sytox image {channel_key}")
            return label_image

        
        elif channel_key == 'ch1':
            # print(composite_image.shape)
            print(f"have processed sytox image {channel_key}")
            return composite_image

    except Exception as e:
        print(f"An error occurred while processing {channel_key} - {region_key}: {e}")
        raise  # Re-raise the exception so the process pool will capture it




def split_images_by_channel(input_dir, output_root_name="channel"):
    input_dir = Path(input_dir).resolve()
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    # Create output root folder if not exists
    output_root = input_dir.parent / output_root_name
    output_root.mkdir(parents=True, exist_ok=True)

    # Get all files in the directory
    files = os.listdir(input_dir)

    # Initialize a dictionary to hold files grouped by (region, ch1 or ch2)
    matching_files_dict = {}

    # Process files
    for file in files:
        match_pattern = pattern.search(file)  # Match r01c01f01 (region pattern)
        match_channel = channel_pattern.search(file)  # Match ch1, ch2 (channel pattern)
        
        if match_pattern and match_channel:
            # Extract the region (e.g., r01c01)
            region_key = match_pattern.group(0)  # e.g., r01c01f01 -> r01c01
            
            # Extract the channel (ch1 or ch2)
            channel_key = match_channel.group(0).lower()  # e.g., ch1 or ch2
            
            # Group by region, then by channel
            key = (region_key, channel_key)
            
            # Add the file to the corresponding key in the dictionary
            if key not in matching_files_dict:
                matching_files_dict[key] = []
            matching_files_dict[key].append(file)
            
    # range all regions
    for region_key in matching_files_dict:
        # Process the ch1 and ch2 files for this region
        ch1_files = [file for file in matching_files_dict.get((region_key[0], 'ch1'), [])]
        ch2_files = [file for file in matching_files_dict.get((region_key[0], 'ch2'), [])]
        
        # print(ch1_files)
        # print(ch2_files)
        # print(region_key)
        
        # processing ch1 files
        sytox_label = process_image_group("ch2", region_key[0], ch1_files, input_dir, output_root)
        # processing ch2 files
        bf_image = process_image_group("ch1", region_key[0], ch2_files, input_dir, output_root)
        
        # print(sytox_label.shape)
        # print(bf_image.shape)
        
        # process the bf image and sytox, to compute the ratio
        # main(region_key, sytox_label, bf_image, output_root)
        
        


# Example usage
if __name__ == "__main__":
    input_dir = r"/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250714_PE/20250713_A2780_24drug_1536_test__2025-07-13T11_24_07-Measurement 1/Images"
    split_images_by_channel(input_dir)
    
    
    



