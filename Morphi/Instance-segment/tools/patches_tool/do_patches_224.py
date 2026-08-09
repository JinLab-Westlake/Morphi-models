import cv2
import os
import numpy as np
from skimage.measure import label, regionprops
import os

def extract_boxes_from_mask(mask):
    """The bounding boxes of all connected regions in the mask are extracted"""
    labeled = label(mask)
    regions = regionprops(labeled)
    boxes = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox  # (y1, x1, y2, x2)
        boxes.append([minc, minr, maxc, maxr])  # 转换成 (x1, y1, x2, y2)
    return np.array(boxes)



def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1+x2) // 2, (y1+y2) // 2])

def is_inside(patch,box):
    cx,cy = box_center(box)
    return patch[0] <= cx <= patch[2] and patch[1] <= cy <= patch[3]


import numpy as np

def group_boxes_from_left_top(
    boxes, 
    min_size=64, 
    max_size=224, 
    image_shape=(1080,1080), 
    step=16
):
    """
    A set of patch coordinates (x1,y1,x2,y2) is returned to ensure that each covered bbox completely falls inside a certain patch.
    - First sort by (y, x) from top left to bottom right, and the seeds grow in turn
    - For each "uncovered" bbox, generate a square patch (side length in [min_size,max_size]) that fully covers it.
    - If the edge is not fit, the window will be clipped back into the graph (still keeping the bbox intact)
    """
    H, W = image_shape[:2]
    boxes = boxes.astype(int).copy()
    n = len(boxes)
    if n == 0:
        return []

    def clamp(v, lo, hi):
        return max(lo, min(v, hi))

    def bbox_inside(patch, box):
        px1, py1, px2, py2 = patch
        x1,  y1,  x2,  y2  = box
        return (x1 >= px1) and (y1 >= py1) and (x2 <= px2) and (y2 <= py2)

    def make_patch_cover_box(b):
        x1, y1, x2, y2 = map(int, b)
        bw, bh = x2 - x1, y2 - y1
        side = max(bw, bh, min_size)
        side = min(side, max_size)

        # The bbox center is placed as a baseline
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        px1 = clamp(cx - side // 2, 0, max(0, W - side))
        py1 = clamp(cy - side // 2, 0, max(0, H - side))
        px2, py2 = px1 + side, py1 + side

        if x1 < px1:
            shift = px1 - x1
            px1 = clamp(px1 - shift, 0, max(0, W - side)); px2 = px1 + side
        if x2 > px2:
            shift = x2 - px2
            px1 = clamp(px1 + shift, 0, max(0, W - side)); px2 = px1 + side

        if y1 < py1:
            shift = py1 - y1
            py1 = clamp(py1 - shift, 0, max(0, H - side)); py2 = py1 + side
        if y2 > py2:
            shift = y2 - py2
            py1 = clamp(py1 + shift, 0, max(0, H - side)); py2 = py1 + side

        return np.array([px1, py1, px2, py2], dtype=int)

    # Sort from the top left
    order = np.lexsort((boxes[:,0], boxes[:,1]))
    boxes_sorted = boxes[order]
    idx_sorted   = np.arange(n)[order]

    visited = np.zeros(n, dtype=bool)
    patches = []

    for i, b in enumerate(boxes_sorted):
        if visited[i]:
            continue
        # This bbox is used as a seed to generate a patch that can completely cover it
        patch = make_patch_cover_box(b)

        included = []
        for j, bb in enumerate(boxes_sorted):
            if visited[j]:
                continue
            if bbox_inside(patch, bb):
                visited[j] = True
                included.append(idx_sorted[j])

        patches.append(patch)

    return patches






import sys
import os

# Default input
default_mask_folder = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250801/20250801_A2780_NO2_P05-P08_2025-08-01T10_32_09-Measurement_1/Combined/masks"
default_image_folder = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250801/20250801_A2780_NO2_P05-P08_2025-08-01T10_32_09-Measurement_1/Combined/channel_2"


# Read the arguments from the command line
mask_folder = sys.argv[1] if len(sys.argv) > 1 else default_mask_folder
image_folder = sys.argv[2] if len(sys.argv) > 2 else default_image_folder

# if have image folder path, then creat patches folder directly
output_folder = os.path.join(image_folder,"224_patches_New/masks")
output_bright = os.path.join(image_folder,"224_patches_New/brightfiled")

# Make sure paths exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_bright, exist_ok=True)

# check folder path
print(f'mask folder is: {output_folder}')
print(f'bright folder is: {output_bright}')

for filename in os.listdir(mask_folder):
    if not filename.endswith(".png") and not filename.endswith(".tif"):
        continue

    print(f"Processing {filename}")

    # read mask
    mask_path = os.path.join(mask_folder, filename)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Cannot read mask {filename}")
        continue
    
    # print(filename)
    prefix = filename.split('_')[0]
    print(prefix)

    # Extract the connected components box
    boxes = extract_boxes_from_mask(mask)

    patch_coords = group_boxes_from_left_top(boxes, min_size=224, max_size=224, image_shape=mask.shape)

    # Read the original image and the ch1 image
    image = cv2.imread(os.path.join(image_folder, prefix+'.tiff'))
    
    if image is None :
        print(f"Warning: Missing image for {filename}")
        continue

    # Do patch
    for i, (x1, y1, x2, y2) in enumerate(patch_coords):
        patch = image[y1:y2, x1:x2]
        # ch1_patch = ch1_image[y1:y2, x1:x2]
        mask_patch = mask[y1:y2, x1:x2]

        patch_name = f"{os.path.splitext(prefix)[0]}_patch_{i}.png"
        ch1_patch_name = f"{os.path.splitext(prefix)[0]}_patch_{i}.png"

        cv2.imwrite(os.path.join(output_bright, patch_name), patch)
        cv2.imwrite(os.path.join(output_folder, ch1_patch_name), mask_patch)

        # save patch locations
        with open(os.path.join(output_folder, "patch_coords.txt"), "a") as f:
            f.write(f"{patch_name},{x1},{y1},{x2},{y2}\n")

    print(f"Saved {len(patch_coords)} patches for {filename}")

print("All done.")