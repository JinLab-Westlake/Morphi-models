# -*- coding: utf-8 -*-

import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import os
import tifffile as tiff

import sys

import argparse
from detectron2.config import get_cfg

from ...swinb.config import add_swinb_config
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np





class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, confidence_threshold=0.0, min_size=0):
        """
        Args:
            image (np.ndarray): (H, W, C), BGR format.
            confidence_threshold (float): confidence threshold.
            min_size (int): minimal width and height or area of boxes to keep.
        Returns:
            predictions, vis_output, instances
        """
        vis_output = None
        predictions = self.predictor(image)
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)

                # confidence threshold
                high_confidence_mask = instances.scores > confidence_threshold
                instances = instances[high_confidence_mask]

                # Mask IoU-based suppression
                def compute_mask_iou(mask1, mask2):
                    intersection = (mask1 & mask2).sum().item()
                    union = (mask1 | mask2).sum().item()
                    return intersection / union if union > 0 else 0

                # IoU-based suppression
                def compute_iou(box1, box2):
                    x1, y1, x2, y2 = box1
                    x3, y3, x4, y4 = box2
                    xi1, yi1 = max(x1, x3), max(y1, y3)
                    xi2, yi2 = min(x2, x4), min(y2, y4)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (x4 - x3) * (y4 - y3)
                    union_area = area1 + area2 - inter_area
                    return inter_area / union_area if union_area > 0 else 0

                # Get all boxes and scores
                masks = instances.pred_masks.cpu() 
                scores = instances.scores.cpu().numpy()
                keep = []

                for i in range(len(masks)):
                    if i in keep:
                        continue
                    discard = False
                    for j in keep:
                        iou = compute_mask_iou(masks[i], masks[j])
                        if iou > 0.1:
                            if scores[i] <= scores[j]:
                                discard = True
                                break
                    if not discard:
                        keep.append(i)

                # # Apply IoU-based keep
                instances = instances[keep]
                
                
                # Remove boxes too close to image borders
                H, W = image.shape[:2]
                margin = 0

                final_keep = []
                boxes = instances.pred_boxes.tensor.cpu().numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    if (
                        x1 > margin and y1 > margin and
                        x2 < (W - margin) and y2 < (H - margin)
                    ):
                        final_keep.append(i)

                instances = instances[final_keep]
                
                
                # Visualize
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
                

        return predictions, vis_output, instances

    
    

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break



def get_patch_window(start_box, patch_size = 224, image_size = 1980):
    x1, y1 = start_box[:2]
    # make sure patch not out of image
    x1 = int(min(x1, image_size - patch_size))
    y1 = int(min(y1, image_size - patch_size))
    return np.array([x1, y1, x1 + patch_size, y1 + patch_size])

def box_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1+x2) // 2, (y1+y2) // 2])

def is_inside(patch,box):
    cx,cy = box_center(box)
    return patch[0] <= cx <= patch[2] and patch[1] <= cy <= patch[3]    
    
def group_boxes_from_left_top(boxes, patch_size = 224, image_size = 1980):
    boxes = boxes.copy()
    indices = np.arange(len(boxes))
    visited = np.zeros(len(boxes), dtype = bool)
    
    # from the left top side of the object in image
    sort_idx = np.lexsort((boxes[:,0], boxes[:,1]))
    boxes = boxes[sort_idx]
    indices = indices[sort_idx]
    
    patches = []
    
    for i, box in enumerate(boxes):
        if visited[i]:
            continue
        patch = get_patch_window(box, patch_size, image_size)
        included = []
        
        for j, other_box in enumerate(boxes):
            if visited[j]:
                continue
            if is_inside(patch, other_box):
                included.append(indices[j])
                visited[j] = True

        patches.append(patch)

    return patches



cfg = get_cfg()
add_swinb_config(cfg) 

cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_16k_260204/config.yaml")
cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_16k_260204/model_0165599.pth"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg.freeze()

# Initialize the tool
demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)

# Input and output directories
image_file_path = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/IHC_segmentation/Final_ihc_data/ihc_first"
output_file_path = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/IHC_segmentation/Final_ihc_data/ihc_first/OUTPUT"

# argument read from the command line
input_folder = sys.argv[1] if len(sys.argv) > 1 else image_file_path
output_folder = sys.argv[2] if len(sys.argv) > 2 else output_file_path

print(f"✅ Input folder: {input_folder}")
print(f"✅ Output folder: {output_folder}")

os.makedirs(output_folder, exist_ok=True)

# creat mask folder
mask_folder = os.path.join(output_folder, "mask")
os.makedirs(mask_folder, exist_ok=True)

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".tif", ".jpeg",".tiff")) and not filename.startswith('.'):  # Process image files only
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read image
        image = cv2.imread(input_path)

        # Run inference
        predictions, vis_output, instances = demo.run_on_image(image)
        

        # ==== make mask images ====
        h, w = image.shape[:2]
        label_mask = np.zeros((h, w), dtype=np.uint8)
        
        if instances is not None and len(instances) > 0:
            masks = instances.pred_masks.numpy()
            scores = instances.scores.cpu().numpy()

            order = np.argsort(-scores)
            masks = masks[order]

            for idx, m in enumerate(masks, start=1):
                label_mask[m] = idx

            # Save as a 16-bit TIFF file
            mask_name = f"{os.path.splitext(filename)[0]}_label.tif"
            tiff.imwrite(os.path.join(mask_folder, mask_name), label_mask * 255)
            
            
        else:
            merged_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            mask_name = f"{os.path.splitext(filename)[0]}_mask.png"
            tiff.imwrite(os.path.join(mask_folder, mask_name), merged_mask * 255)

        boxes = instances.pred_boxes.tensor.cpu().numpy()

        # Save the result if detection exists
        if vis_output:
            vis_image = vis_output.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV
            cv2.imwrite(output_path, vis_image)
            print(f"Processed: {filename} -> Saved to {output_path}")
        else:
            print(f"No detections for {filename}")


print("Batch processing completed.")


