import numpy as np
import cv2
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

def calculate_sobel_sharpness(image):

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # print(sobel_x)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobel_x, sobel_y)
    return sobel

def process_image_group(key, matching_files, image_file_path, output_file_path):
    if len(matching_files) < 1:
        print(f"Not enough images for {key}")
        return

    # Sort by name to make sure the order is accurate
    matching_files.sort()
    sharpness_stack = []

    image_stack = [cv2.imread(os.path.join(image_file_path, file), cv2.IMREAD_UNCHANGED) for file in matching_files]
    for image in image_stack:
        sharpness = calculate_sobel_sharpness(image)
        sharpness_stack.append(sharpness)

    # Converting to an array
    sharpness_stack = np.array(sharpness_stack)
    image_stack = np.array(image_stack)

    # The index of the layer with the clearest pixel per pixel is obtained
    sharpest_indices = np.argmax(sharpness_stack, axis=0)

    # creat combined image
    height, width = sharpest_indices.shape
    composite_image = np.zeros_like(image_stack[0])

    for i in range(height):
        for j in range(width):
            composite_image[i, j] = image_stack[sharpest_indices[i, j], i, j]

    # Saving composite images
    cv2.imwrite(f"{output_file_path}/{key}.tif", composite_image)
    print(f"Processed {key}")


def main():
    image_file_path = "/Volumes/Expansion/Flowchart_materials/Figure3_Sobel_operator/Died_example/r02c05/bf/IMAGE"
    output_file_path = "/Volumes/Expansion/Flowchart_materials/Figure3_Sobel_operator/Died_example/r02c05/bf/IMAGE/combined"

    os.makedirs(output_file_path, exist_ok=True)

    pattern = re.compile(r"r\d{2}c\d{2}f\d{2}")

    # pattern = re.compile(r"[A-Z]\d{2}.[a-z]\d{1}_T\d{4}F\d{3}L\d{2}")

    files = os.listdir(image_file_path)

    matching_files_dict = {}
    for file in files:
        # print(file)
        match = pattern.search(file)
        if match:
            # print('1')
            key = match.group()
            # print(key)
            if key not in matching_files_dict:
                matching_files_dict[key] = []
            if file.startswith('._'):  # ignore useless file
                continue
            matching_files_dict[key].append(file)

    # use ProcessPoolExecutor to process
    with ProcessPoolExecutor(max_workers=16) as executor:  # Maximum of 16 parallel tasks
        # print('1')
        futures = []
        for key, matching_files in matching_files_dict.items():
            futures.append(executor.submit(process_image_group, key, matching_files, image_file_path, output_file_path))
            print(key)
        # waiting for all task finished
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()