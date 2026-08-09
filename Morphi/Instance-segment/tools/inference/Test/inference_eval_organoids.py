# # -*- coding: utf-8 -*-

import os
import sys
import json
import cv2
import torch
import tifffile as tiff
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from skimage import measure
import pandas as pd


sys.path.insert(0, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main")

from swinb.config import add_swinb_config

# ---------------------- COCO Instance Metrics ----------------------
def mask_to_rle(mask):
    """Convert binary mask to COCO RLE"""
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle

# def seg_to_rle(segmentation, height, width):
#     """Convert COCO segmentation (polygon or RLE) to RLE"""
#     if isinstance(segmentation, list):
#         rle = maskUtils.frPyObjects(segmentation, height, width)
#         rle = maskUtils.merge(rle)
#     elif isinstance(segmentation, dict) and 'counts' in segmentation:
#         rle = segmentation
#         if isinstance(rle['counts'], bytes):
#             rle['counts'] = rle['counts'].decode('utf-8')
#     else:
#         raise ValueError("Unknown segmentation type")
#     return rle


def seg_to_rle(segmentation, height, width):
    """
    Convert COCO segmentation (polygon or RLE) to RLE safely
    """
    if isinstance(segmentation, list):
        if len(segmentation) == 0:
            # 空 polygon
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            return mask_to_rle(empty_mask)
        try:
            rle = maskUtils.frPyObjects(segmentation, height, width)
            rle = maskUtils.merge(rle)
        except Exception as e:
            # fallback: empty mask
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            rle = mask_to_rle(empty_mask)
    elif isinstance(segmentation, dict) and 'counts' in segmentation:
        rle = segmentation
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
    else:
        # fallback empty mask
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        rle = mask_to_rle(empty_mask)
    return rle

# def evaluate_coco_instance(pred_masks, gt_annotations, height, width, iou_thresh=0.5):
#     """Compute TP, FP, FN, precision, recall, F1"""
#     gt_ids = list(range(len(gt_annotations)))
#     matched_gt = set()
#     TP = 0
#     FP = 0

#     for pred_mask in pred_masks:
#         pred_rle = mask_to_rle(pred_mask)
#         best_iou = 0
#         best_gt_idx = None
#         for gt_idx in gt_ids:
#             if gt_idx in matched_gt:
#                 continue
#             gt_seg = gt_annotations[gt_idx]['segmentation']
#             gt_rle = seg_to_rle(gt_seg, height, width)
#             iou = maskUtils.iou([pred_rle], [gt_rle], [0])[0][0]
#             if iou > best_iou:
#                 best_iou = iou
#                 best_gt_idx = gt_idx
#         if best_iou >= iou_thresh and best_gt_idx is not None:
#             TP += 1
#             matched_gt.add(best_gt_idx)
#         else:
#             FP += 1

#     FN = len(gt_annotations) - len(matched_gt)
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#     return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "f1": f1}

def evaluate_coco_instance(pred_masks, gt_annotations, height, width, iou_thresh=0.5):
    """Compute TP, FP, FN, precision, recall, F1"""
    gt_ids = list(range(len(gt_annotations)))
    matched_gt = set()
    TP = 0
    FP = 0

    for pred_mask in pred_masks:
        # 确保 pred_mask 与 GT mask shape 一致
        if pred_mask.shape != (height, width):
            pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            pred_mask_resized = pred_mask_resized.astype(bool)
        else:
            pred_mask_resized = pred_mask

        pred_rle = mask_to_rle(pred_mask_resized)
        best_iou = 0
        best_gt_idx = None
        for gt_idx in gt_ids:
            if gt_idx in matched_gt:
                continue
            gt_seg = gt_annotations[gt_idx]['segmentation']
            gt_rle = seg_to_rle(gt_seg, height, width)
            iou = maskUtils.iou([pred_rle], [gt_rle], [0])[0][0]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_thresh and best_gt_idx is not None:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    FN = len(gt_annotations) - len(matched_gt)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "f1": f1}



def overlay_mask_on_image(image, label_mask):
    """
    image: 原图 (H, W, 3) BGR
    label_mask: HxW uint16，每个实例为不同整数ID
    """
    overlay = image.copy()
    num_instances = label_mask.max()
    if num_instances == 0:
        return overlay

    colors = np.random.randint(0, 255, size=(num_instances + 1, 3), dtype=np.uint8)

    for idx in range(1, num_instances + 1):
        mask = label_mask == idx
        if np.any(mask):
            # 创建和图像同尺寸的颜色图
            color_img = np.zeros_like(overlay, dtype=np.uint8)
            color_img[mask] = colors[idx]  # 只填充当前 mask 区域
            # 叠加当前 mask 区域
            overlay = cv2.addWeighted(overlay, 1.0, color_img, 0.5, 0)

    return overlay

import cv2
import numpy as np
from pycocotools import mask as maskUtils

def overlay_mask_with_iou(image, pred_masks, gt_annotations, height, width):
    """
    在原图上叠加每个预测 mask 并显示 IoU，GT 用黄线表示。
    
    Args:
        image: 原图 (H,W,3) BGR
        pred_masks: list of np.ndarray (H,W) bool, 每个预测 mask
        gt_annotations: list of COCO annotations
        height, width: 原图尺寸
    Returns:
        overlay_img: 图像叠加结果
    """
    overlay = image.copy()
    colors = np.random.randint(0, 255, size=(len(pred_masks), 3), dtype=np.uint8)
    
    for idx, pred_mask in enumerate(pred_masks):
        # resize mask到原图尺寸
        if pred_mask.shape != (height, width):
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # 找最佳匹配 GT
        best_iou = 0
        best_gt_mask = None
        for gt_ann in gt_annotations:
            gt_rle = gt_ann['segmentation']
            gt_mask = maskUtils.decode(seg_to_rle(gt_rle, height, width)).astype(bool)
            iou = maskUtils.iou([maskUtils.encode(np.asfortranarray(pred_mask.astype(np.uint8)))],
                                [maskUtils.encode(np.asfortranarray(gt_mask.astype(np.uint8)))], [0])[0][0]
            if iou > best_iou:
                best_iou = iou
                best_gt_mask = gt_mask

        # 叠加预测 mask
        color_img = np.zeros_like(overlay, dtype=np.uint8)
        color_img[pred_mask] = colors[idx]
        overlay = cv2.addWeighted(overlay, 1.0, color_img, 0.5, 0)
        
        # 在 mask 上显示 IoU
        yx = np.argwhere(pred_mask)
        if yx.shape[0] > 0:
            y, x = yx.mean(axis=0).astype(int)
            cv2.putText(overlay, f"{best_iou:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1, cv2.LINE_AA)
        
        # 画 GT 轮廓
        if best_gt_mask is not None:
            contours, _ = cv2.findContours(best_gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0,255,255), 1)  # 黄线 BGR=(0,255,255)

    return overlay



# ---------------------- Visualization Demo ----------------------
class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image,confidence_threshold = 0.6):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
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

                # Step 1: confidence threshold
                high_confidence_mask = instances.scores > confidence_threshold
                instances = instances[high_confidence_mask]
                
                
                # Step 2: Mask IoU-based suppression
                def compute_mask_iou(mask1, mask2):
                    intersection = (mask1 & mask2).sum().item()
                    union = (mask1 | mask2).sum().item()
                    return intersection / union if union > 0 else 0

                # Step 2: IoU-based suppression
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
                # boxes = instances.pred_boxes.tensor.cpu().numpy()
                masks = instances.pred_masks.cpu() 
                scores = instances.scores.cpu().numpy()
                keep = []

                for i in range(len(masks)):
                    if i in keep:
                        continue
                    discard = False
                    for j in keep:
                        iou = compute_mask_iou(masks[i], masks[j])
                        if iou > 0.01:
                            if scores[i] <= scores[j]:
                                discard = True
                                break
                    if not discard:
                        keep.append(i)

                # 只保留非冗余实例
                instances = instances[keep]

                
                # # Step 3: Remove boxes too close to image borders
                # H, W = image.shape[:2]
                # margin = -1  # 可调整：距离边缘小于5像素的 box 被移除

                # final_keep = []
                # boxes = instances.pred_boxes.tensor.cpu().numpy()

                # for i, box in enumerate(boxes):
                #     x1, y1, x2, y2 = box
                #     if (
                #         x1 > margin and y1 > margin and
                #         x2 < (W - margin) and y2 < (H - margin)
                #     ):
                #         final_keep.append(i)

                # instances = instances[final_keep]
                
        
                # Visualize
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
                
                

        return predictions, vis_output,instances

# # ---------------------- Main ----------------------
# def main(input_folder, output_folder, gt_json_file, confidence_threshold=0.0, iou_thresh=0.5):
#     coco_gt = COCO(gt_json_file)

#     cfg = get_cfg()
#     add_swinb_config(cfg)
#     cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_16k/config.yaml")
#     cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_16k/model_0021049.pth"
#     cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     cfg.freeze()

#     demo = VisualizationDemo(cfg)

#     os.makedirs(output_folder, exist_ok=True)
#     mask_folder = os.path.join(output_folder, "mask")
#     os.makedirs(mask_folder, exist_ok=True)

#     total_metrics = {"TP":0, "FP":0, "FN":0}

#     for filename in os.listdir(input_folder):
#         if not filename.lower().endswith((".jpg", ".png", ".tif", ".tiff", ".jpeg")):
#             continue
#         if filename.startswith('._'):
#             continue
        
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)

#         # Read image
#         try:
#             image = cv2.imread(input_path)
#             if image is None:
#                 image = tiff.imread(input_path)
#                 if image.ndim == 2:
#                     image = np.stack([image]*3, axis=-1)
#         except Exception as e:
#             print(f"Failed to read {input_path}: {e}")
#             continue

#         predictions, vis_output, instances = demo.run_on_image(image, confidence_threshold)

#         # h, w = image.shape[:2]
#         # label_mask = np.zeros((h, w), dtype=np.uint16)
#         # pred_masks = []
#         # if instances is not None and len(instances) > 0:
#         #     masks = instances.pred_masks.cpu().numpy().astype(bool)
#         #     scores = instances.scores.cpu().numpy()
#         #     order = np.argsort(-scores)
#         #     masks = masks[order]
#         #     for idx, m in enumerate(masks, start=1):
#         #         label_mask[m] = idx
#         #         pred_masks.append(m)
        
#         h, w = image.shape[:2]  # 原图尺寸
#         pred_masks = []
#         label_mask = np.zeros((h, w), dtype=np.uint16)
#         if instances is not None and len(instances) > 0:
#             masks = instances.pred_masks.cpu().numpy().astype(bool)
#             scores = instances.scores.cpu().numpy()
#             order = np.argsort(-scores)
#             masks = masks[order]
#             for idx, m in enumerate(masks, start=1):
#                 # resize预测mask到原图尺寸
#                 if m.shape != (h, w):
#                     m_resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
#                 else:
#                     m_resized = m
#                 label_mask[m_resized] = idx
#                 pred_masks.append(m_resized)

#         mask_name = os.path.join(mask_folder, f"{os.path.splitext(filename)[0]}_label.tif")
#         tiff.imwrite(mask_name, label_mask)
        
        
#         # # # overlay mask 回原图
#         # overlay_img = overlay_mask_on_image(image, label_mask)
#         # overlay_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_overlay.png")
#         # cv2.imwrite(overlay_path, overlay_img)
        


#         if vis_output:
#             vis_img = vis_output.get_image()[:, :, ::-1]
#             cv2.imwrite(output_path, vis_img)

#         # Match COCO image by file_name
#         img_id_list = [img_id for img_id, img_info in coco_gt.imgs.items()
#                        if img_info['file_name'] == filename]
#         if not img_id_list:
#             print(f"No COCO GT for {filename}")
#             continue
#         img_id = img_id_list[0]
#         img_info = coco_gt.imgs[img_id]
#         height, width = img_info['height'], img_info['width']

#         ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
#         gt_anns = coco_gt.loadAnns(ann_ids)

#         metrics = evaluate_coco_instance(pred_masks, gt_anns, height, width, iou_thresh)
#         print(f"{filename}: {metrics}")
        
        
#         overlay_img = overlay_mask_with_iou(
#             image,
#             pred_masks,
#             gt_anns,
#             height=image.shape[0],
#             width=image.shape[1]
#         )
#         overlay_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_overlay.png")
#         cv2.imwrite(overlay_path, overlay_img)

#         for k in total_metrics:
#             total_metrics[k] += metrics[k]

#     TP = total_metrics["TP"]
#     FP = total_metrics["FP"]
#     FN = total_metrics["FN"]
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#     print("===== Overall Metrics =====")
#     print(f"TP: {TP}, FP: {FP}, FN: {FN}")
#     print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


# # ================= Command-line Interface =================
# if __name__ == "__main__":
#     if len(sys.argv) < 4:
#         print("Usage: python run_inference_coco_eval.py <input_folder> <output_folder> <gt_json_file> [confidence_threshold] [iou_threshold]")
#         sys.exit(1)

#     input_folder = sys.argv[1]
#     output_folder = sys.argv[2]
#     gt_json_file = sys.argv[3]
#     confidence_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
#     iou_thresh = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

#     main(input_folder, output_folder, gt_json_file, confidence_threshold, iou_thresh)



from pathlib import Path


def image_key(name):
    """
    把 xxx.png / xxx.tif / xxx.jpeg.tiff 统一成同一个 key。
    用于匹配 input image 和 COCO json 里的 file_name。
    """
    name = os.path.basename(name)

    image_exts = {
        ".png", ".jpg", ".jpeg", ".tif", ".tiff",
        ".bmp", ".svs"
    }

    p = Path(name)

    # 连续去扩展名，例如 xxx.jpeg.tiff -> xxx
    while p.suffix.lower() in image_exts:
        p = Path(p.stem)

    return p.name





def evaluate_for_confidence(input_folder, output_folder, gt_json_file, cfg, confidence_threshold=0.5, iou_thresh=0.5):
    coco_gt = COCO(gt_json_file)
    demo = VisualizationDemo(cfg)

    os.makedirs(output_folder, exist_ok=True)
    mask_folder = os.path.join(output_folder, "mask")
    os.makedirs(mask_folder, exist_ok=True)

    per_image_metrics = []
    total_metrics = {"TP":0, "FP":0, "FN":0}

    coco_name_to_id = {}
    for img_id, img_info in coco_gt.imgs.items():
        key = image_key(img_info["file_name"])
        
        if key in coco_name_to_id:
            print(f"Warning: duplicated coco key: {key}")
        
        coco_name_to_id[key] = img_id
    
    
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".jpg", ".png", ".tif", ".tiff", ".jpeg")):
            continue
        if filename.startswith('._'):
            continue
        
        input_path = os.path.join(input_folder, filename)

        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            image = tiff.imread(input_path)
            if image.ndim == 2:
                image = np.stack([image]*3, axis=-1)

        predictions, vis_output, instances = demo.run_on_image(image, confidence_threshold)

        h, w = image.shape[:2]
        pred_masks = []
        label_mask = np.zeros((h, w), dtype=np.uint16)
        if instances is not None and len(instances) > 0:
            masks = instances.pred_masks.cpu().numpy().astype(bool)
            scores = instances.scores.cpu().numpy()
            order = np.argsort(-scores)
            masks = masks[order]
            for idx, m in enumerate(masks, start=1):
                if m.shape != (h, w):
                    m_resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    m_resized = m
                label_mask[m_resized] = idx
                pred_masks.append(m_resized)

        
        h, w = image.shape[:2]
        pred_masks = []
        label_mask = np.zeros((h, w), dtype=np.uint16)

        if instances is not None and len(instances) > 0:
            masks = instances.pred_masks.cpu().numpy().astype(bool)
            scores = instances.scores.cpu().numpy()

            order = np.argsort(-scores)
            masks = masks[order]

            for idx, m in enumerate(masks, start=1):
                if m.shape != (h, w):
                    m_resized = cv2.resize(
                        m.astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    m_resized = m

                label_mask[m_resized] = idx
                pred_masks.append(m_resized)

        # ==========================
        # 保存推理 mask：放在这里
        # ==========================
        conf_mask_folder = os.path.join(
            output_folder,
            "mask",
            f"conf_{confidence_threshold:.2f}"
        )
        os.makedirs(conf_mask_folder, exist_ok=True)

        mask_name = os.path.join(
            conf_mask_folder,
            f"{os.path.splitext(filename)[0]}_label.tif"
        )

        tiff.imwrite(mask_name, label_mask.astype(np.uint16))
        print(f"Saved inference mask: {mask_name}")
        
        
        
        # 先匹配 COCO image id
        key = image_key(filename)
        img_id = coco_name_to_id.get(key, None)

        print(f"input filename: {filename}")
        print(f"match key: {key}")
        print(f"matched coco img_id: {img_id}")

        if img_id is None:
            print(f"SKIP: no COCO GT matched for {filename}")
            continue

        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))

        print(
            f"{filename}: matched COCO file = {coco_gt.imgs[img_id]['file_name']}, "
            f"GT anns = {len(gt_anns)}, pred masks = {len(pred_masks)}"
        )

        # 计算 metrics
        metrics = evaluate_coco_instance(pred_masks, gt_anns, h, w, iou_thresh)

        metrics_row = {
            "filename": filename,
            "matched_coco_file": coco_gt.imgs[img_id]["file_name"],
            "match_key": key,
            "confidence": confidence_threshold,
        }
        metrics_row.update(metrics)
        per_image_metrics.append(metrics_row)

        for k in total_metrics:
            total_metrics[k] += metrics[k]

        # 再画 overlay
        overlay_img = overlay_mask_with_iou(
            image=image,
            pred_masks=pred_masks,
            gt_annotations=gt_anns,
            height=h,
            width=w
        )

        overlay_path = os.path.join(
            output_folder,
            f"{os.path.splitext(filename)[0]}_overlay_conf{confidence_threshold:.2f}.png"
        )
        cv2.imwrite(overlay_path, overlay_img)
        
        
        
        
        # # 保存 mask
        # mask_name = os.path.join(mask_folder, f"{os.path.splitext(filename)[0]}_label.tif")
        # tiff.imwrite(mask_name, label_mask)

        # # 保存 overlay
        # overlay_img = overlay_mask_with_iou(
        #     image, pred_masks, 
        #     # gt_annotations=coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id for img_id, info in coco_gt.imgs.items() if info['file_name']==filename])),
        #     height=h, width=w
        # )
        
        # overlay_img = overlay_mask_with_iou(
        #     image=image,
        #     pred_masks=pred_masks,
        #     gt_annotations=gt_anns,
        #     height=h,
        #     width=w
        # )
        
        # overlay_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_overlay_conf{confidence_threshold:.2f}.png")
        # cv2.imwrite(overlay_path, overlay_img)

        # COCO metric
        # img_id_list = [img_id for img_id, img_info in coco_gt.imgs.items()
        #                if img_info['file_name'] == filename]
        # if not img_id_list:
        #     continue
        # img_id = img_id_list[0]
        # gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))

        # metrics = evaluate_coco_instance(pred_masks, gt_anns, h, w, iou_thresh)
        # metrics_row = {"filename": filename, "confidence": confidence_threshold}
        # metrics_row.update(metrics)
        # per_image_metrics.append(metrics_row)

        # for k in total_metrics:
        #     total_metrics[k] += metrics[k]

    TP = total_metrics["TP"]
    FP = total_metrics["FP"]
    FN = total_metrics["FN"]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    overall_metrics = {"confidence": confidence_threshold, "TP":TP, "FP":FP, "FN":FN,
                       "precision": precision, "recall": recall, "f1": f1}

    return per_image_metrics, overall_metrics


# ====================== 循环不同 confidence ======================
if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    gt_json_file = sys.argv[3]
    iou_thresh = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    cfg = get_cfg()
    add_swinb_config(cfg)
    
    """
    generalized weight
    """
    # cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_16k_260204/config.yaml")
    # cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_16k_260204/model_0165599.pth"
    
    """
    organoids fintuned weight about 20 epoch
    """
    cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/config.yaml")
    cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO/model_0001049.pth"
    
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    all_image_metrics = []
    all_overall_metrics = []
    


    conf_thresholds = np.arange(0.05, 1, 0.05)
    for conf in conf_thresholds:
        print(f"Evaluating confidence threshold: {conf}")
        per_img, overall = evaluate_for_confidence(input_folder, output_folder, gt_json_file, cfg, conf, iou_thresh)
        all_image_metrics.extend(per_img)
        all_overall_metrics.append(overall)

    # 保存 Excel
    image_df = pd.DataFrame(all_image_metrics)
    overall_df = pd.DataFrame(all_overall_metrics)
    excel_path = os.path.join(output_folder, "evaluation_metrics.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        image_df.to_excel(writer, sheet_name="PerImage", index=False)
        overall_df.to_excel(writer, sheet_name="Overall", index=False)

    print(f"Saved evaluation metrics to {excel_path}")