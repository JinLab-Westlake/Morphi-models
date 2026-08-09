# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import os
import sys
import ast
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import tifffile as tiff
# tiff.imwrite(os.path.join(mask_folder, mask_name), label_mask)

# Add your custom path
sys.path.insert(0, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main")
from swinb.config import add_swinb_config

class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, confidence_threshold=0.0, excluded_areas=[]):
        """
        Processes a single image. 
        - excluded_areas: coordinates found in ANY frame that should be blocked.
        """
        vis_output = None
        large_boxes = [] 
        predictions = self.predictor(image)
        image_rgb = image[:, :, ::-1]
        visualizer = Visualizer(image_rgb, self.metadata, instance_mode=self.instance_mode)

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)

            # --- Step 1: Confidence threshold ---
            instances = instances[instances.scores > confidence_threshold]

            # --- Step 2: IoU-based suppression (Original Logic) ---
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

            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            keep = []
            for i in range(len(boxes)):
                if i in keep: continue
                discard = False
                for j in keep:
                    if compute_iou(boxes[i], boxes[j]) > 0.05:
                        if scores[i] <= scores[j]:
                            discard = True
                            break
                if not discard:
                    keep.append(i)
            instances = instances[keep]

            # --- Step 3: Global Exclusion Check (Remove if overlaps with blocked area) ---
            if len(instances) > 0 and len(excluded_areas) > 0:
                current_boxes = instances.pred_boxes.tensor.cpu().numpy()
                keep_after_exclusion = []
                for box in current_boxes:
                    is_excluded = False
                    for ex_box in excluded_areas:
                        # Spatial intersection check
                        if not (box[2] < ex_box[0] or box[0] > ex_box[2] or 
                                box[3] < ex_box[1] or box[1] > ex_box[3]):
                            is_excluded = True
                            break
                    keep_after_exclusion.append(not is_excluded)
                instances = instances[keep_after_exclusion]

            # --- Step 4: Size Filtering and Recording ---
            if len(instances) > 0:
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                # Record boxes > 2000 to be blocked in ALL masks
                large_boxes = boxes[sizes > 6000]
                
                # Keep only boxes <= 2000 for current frame result
                instances = instances[sizes <= 6000]

            vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output, instances, large_boxes

# ========= Configuration =========
cfg = get_cfg()
add_swinb_config(cfg) 
cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_6k/config.yaml")
cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_6k/model_0011699.pth"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.freeze()

demo = VisualizationDemo(cfg)

# ========= Paths =========
image_file_path = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/Test_dataset/20251124FISH_cold_16d/2z_ch1_frames_modified/denoised"
output_file_path = os.path.join(image_file_path, "output")

input_folder = sys.argv[1] if len(sys.argv) > 1 else image_file_path
output_folder = sys.argv[2] if len(sys.argv) > 2 else output_file_path
mask_folder = os.path.join(output_folder, "mask")

os.makedirs(mask_folder, exist_ok=True)

# List of all valid files
image_files = sorted([f for f in os.listdir(input_folder) 
                     if f.lower().endswith((".jpg", ".png", ".tif", ".jpeg", ".tiff")) 
                     and not f.startswith('.')])

# ========= PASS 1: SCAN ALL FOR LARGE AREAS =========
global_excluded_areas = []
print(f"🚀 Pass 1: Scanning {len(image_files)} images for large artifacts...")

for filename in image_files:
    img = cv2.imread(os.path.join(input_folder, filename))
    if img is None: continue
    
    # Run with empty exclusion to find everything
    _, _, _, new_large_boxes = demo.run_on_image(img, excluded_areas=[])
    
    if len(new_large_boxes) > 0:
        for lb in new_large_boxes:
            global_excluded_areas.append(lb)
            print(f"📍 Large area recorded in {filename}: {lb}")

print(f"✅ Pass 1 complete. Total forbidden areas found: {len(global_excluded_areas)}")

# ========= PASS 2: FINAL INFERENCE & MASK GENERATION =========
print(f"🚀 Pass 2: Generating cleaned results...")

for filename in image_files:
    img = cv2.imread(os.path.join(input_folder, filename))
    if img is None: continue

    # Use the full exclusion list gathered from all frames
    predictions, vis_output, instances, _ = demo.run_on_image(
        img, excluded_areas=global_excluded_areas
    )

    # 1. Save Mask (Only contains areas <= 2000 that don't overlap with blocked areas)
    if instances is not None and len(instances) > 0:
        masks = instances.pred_masks.numpy()
        merged_mask = np.any(masks, axis=0).astype(np.uint8)
        # 2. Save Instance Label Mask (background=0, instance id=1..N)
        h, w = img.shape[:2]
        label_mask = np.zeros((h, w), dtype=np.uint16)  # 使用 uint16 以避免溢出

        if instances is not None and len(instances) > 0:
            masks = instances.pred_masks.cpu().numpy().astype(bool)  # (N,H,W)
            scores = instances.scores.cpu().numpy()

            # 可选：按 score 从高到低排序，让高置信度覆盖低置信度
            order = np.argsort(-scores)
            masks = masks[order]

            for idx, m in enumerate(masks, start=1):  # 从 1 开始给每个实例编号
                label_mask[m] = idx  # 将每个实例的区域设置为其 ID

        # 保存为 16-bit TIFF 文件
        mask_name = f"{os.path.splitext(filename)[0]}_label.tif"
        tiff.imwrite(os.path.join(mask_folder, mask_name), label_mask)
    else:
        merged_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        mask_name = f"{os.path.splitext(filename)[0]}_label.tif"
        tiff.imwrite(os.path.join(mask_folder, mask_name), merged_mask)

    # mask_name = f"{os.path.splitext(filename)[0]}_mask.png"
    # cv2.imwrite(os.path.join(mask_folder, mask_name), merged_mask * 255)

    # 2. Save Visualization
    if vis_output:
        cv2.imwrite(os.path.join(output_folder, filename), vis_output.get_image()[:, :, ::-1])
        print(f"Processed: {filename}")

print(f"✨ All done. Results saved in: {output_folder}")