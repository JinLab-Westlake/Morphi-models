# -*- coding: utf-8 -*-

import os
import sys
import cv2
import torch
import tifffile as tiff
import numpy as np
import pandas as pd

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


from ...swinb.config import add_swinb_config


class VisualizationDemo:
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, confidence_threshold=0.3, min_size=0):
        vis_output = None
        instances = None

        predictions = self.predictor(image)
        image_rgb = image[:, :, ::-1]
        visualizer = Visualizer(image_rgb, self.metadata, instance_mode=self.instance_mode)

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

                # mask IoU suppression
                def compute_mask_iou(mask1, mask2):
                    intersection = (mask1 & mask2).sum().item()
                    union = (mask1 | mask2).sum().item()
                    return intersection / union if union > 0 else 0

                masks = instances.pred_masks.cpu()
                scores = instances.scores.cpu().numpy()
                keep = []

                for i in range(len(masks)):
                    discard = False
                    for j in keep:
                        iou = compute_mask_iou(masks[i], masks[j])
                        if iou > 0.01 and scores[i] <= scores[j]:
                            discard = True
                            break
                    if not discard:
                        keep.append(i)

                instances = instances[keep]

                # remove boxes too close to image borders
                H, W = image_rgb.shape[:2]
                margin = 5
                final_keep = []
                boxes = instances.pred_boxes.tensor.cpu().numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    if x1 > margin and y1 > margin and x2 < (W - margin) and y2 < (H - margin):
                        final_keep.append(i)

                instances = instances[final_keep]
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                size_mask = sizes >= 0
                instances = instances[size_mask]

                # Visualize
                vis_output = image.copy()

        return predictions, vis_output, instances


def find_mask_path(mask_folder, filename_stem):
    """
    Automatically match live/dead masks by filename
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
    Read the live/dead mask
    Ask for the mask pixel value:
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


def process_one_case(input_folder, output_folder, semantic_mask_folder, demo):
    print(f"\n=== Processing case ===")
    print(f"input_folder: {input_folder}")
    print(f"output_folder: {output_folder}")
    print(f"mask_folder: {semantic_mask_folder}")

    os.makedirs(output_folder, exist_ok=True)

    overlay_folder = os.path.join(output_folder, "overlay")
    os.makedirs(overlay_folder, exist_ok=True)

    instance_mask_folder = os.path.join(output_folder, "instance_mask")
    os.makedirs(instance_mask_folder, exist_ok=True)

    xlsx_path = os.path.join(output_folder, "instance_death_rate.xlsx")
    
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
            print(f"Cannot read image: {input_path}")
            continue

        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(overlay_folder, filename)

        semantic_mask_path = find_mask_path(semantic_mask_folder, stem)
        semantic_mask = read_mask(semantic_mask_path)

        if semantic_mask is not None and semantic_mask.shape[:2] != image.shape[:2]:
            semantic_mask = cv2.resize(
                semantic_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        predictions, vis_output, instances = demo.run_on_image(image)

        vis_image = vis_output[:, :, ::-1].copy()  # RGB -> BGR
        h, w = image.shape[:2]
        label_mask = np.zeros((h, w), dtype=np.uint16)
        overlay_annotations = []

        if instances is not None and len(instances) > 0:
            masks = instances.pred_masks.cpu().numpy().astype(bool)
            scores = instances.scores.cpu().numpy()
            boxes = instances.pred_boxes.tensor.cpu().numpy()

            order = np.argsort(-scores)
            masks = masks[order]
            scores = scores[order]
            boxes = boxes[order]

            for idx, m in enumerate(masks, start=1):
                label_mask[m] = idx

            instance_mask_name = f"{stem}_instance_label.tif"
            tiff.imwrite(os.path.join(instance_mask_folder, instance_mask_name), label_mask)

            for idx, (m, score, box) in enumerate(zip(masks, scores, boxes), start=1):
                x1, y1, x2, y2 = box.tolist()

                if semantic_mask is not None:
                    live_overlap = m & (semantic_mask == 127)
                    dead_overlap = m & (semantic_mask == 254)
                    live_pixels = int(np.sum(live_overlap))
                    dead_pixels = int(np.sum(dead_overlap))
                    overlap_pixels = live_pixels + dead_pixels
                    death_rate = dead_pixels / overlap_pixels if overlap_pixels > 0 else np.nan
                else:
                    live_overlap = None
                    dead_overlap = None
                    live_pixels = 0
                    dead_pixels = 0
                    overlap_pixels = 0
                    death_rate = np.nan

                instance_area = int(np.sum(m))

                results.append({
                    "image_name": filename,
                    "instance_id": idx,
                    "instance_area": instance_area,
                    "live_pixels": live_pixels,
                    "dead_pixels": dead_pixels,
                    "overlap_pixels": overlap_pixels,
                    "death_rate": death_rate
                })

                overlay_annotations.append({
                    "instance_id": idx,
                    "x1": int(x1),
                    "y1": int(y1),
                    "live_pixels": live_pixels,
                    "dead_pixels": dead_pixels,
                    "death_rate": death_rate,
                    "live_mask": live_overlap.copy() if live_overlap is not None else None,
                    "dead_mask": dead_overlap.copy() if dead_overlap is not None else None,
                    "instance_mask": m.copy()
                })

        else:
            instance_mask_name = f"{stem}_instance_label.tif"
            tiff.imwrite(os.path.join(instance_mask_folder, instance_mask_name), label_mask)
            print(f"No instances detected in {filename}")

        # overlay: no box, only masks + text
        alpha = 0.2
        for ann in overlay_annotations:
            live_mask = ann["live_mask"]
            dead_mask = ann["dead_mask"]
            instance_mask = ann["instance_mask"]

            if instance_mask is not None:
                vis_image[instance_mask] = (
                    vis_image[instance_mask] * 0.8 + np.array([255, 255, 255]) * 0.2
                ).astype(np.uint8)

            if live_mask is not None:
                vis_image[live_mask] = (
                    vis_image[live_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha
                ).astype(np.uint8)

            if dead_mask is not None:
                vis_image[dead_mask] = (
                    vis_image[dead_mask] * (1 - alpha) + np.array([0, 0, 255]) * alpha
                ).astype(np.uint8)

        for ann in overlay_annotations:
            x1, y1 = ann["x1"], ann["y1"]
            live_pixels = ann["live_pixels"]
            dead_pixels = ann["dead_pixels"]
            death_rate = ann["death_rate"]
            instance_id = ann["instance_id"]

            if np.isnan(death_rate):
                text = f"ID{instance_id} - DR:NA"
                color = (255, 255, 255)
            else:
                text = f"ID{instance_id} - DR:{death_rate:.2f}"
                color = (0, 255, 0) if death_rate < 0.33 else ((0, 255, 255) if death_rate < 0.66 else (0, 0, 255))

            text_x = max(0, x1)
            text_y = max(15, y1 - 5)

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis_image, (text_x, text_y - th - 4), (text_x + tw + 2, text_y + 2), (0, 0, 0), -1)
            cv2.putText(vis_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.imwrite(output_path, vis_image)
        print(f"Saved overlay: {output_path}")

    df = pd.DataFrame(results)
    df.to_excel(xlsx_path, index=False)
    print(f"Saved xlsx: {xlsx_path}")


def find_case_folders_in_root_a(root_a):
    """
    Find all root_a/<case_name>/channel/ch1
    :return [(case_name, input_folder, output_folder), ...]
    """
    cases = []
    for name in sorted(os.listdir(root_a)):
        case_dir = os.path.join(root_a, name)
        if not os.path.isdir(case_dir):
            continue

        input_folder = os.path.join(case_dir, "channel", "more_bf")
        output_folder = os.path.join(case_dir,"channel", "output_r02c02")

        if os.path.isdir(input_folder):
            cases.append((name, input_folder, output_folder))
    return cases


def batch_process(root_a, root_b, demo):
    cases = find_case_folders_in_root_a(root_a)
    print(f"Found {len(cases)} candidate cases in root_a")

    matched = 0
    for case_name, input_folder, output_folder in cases:

        semantic_mask_folder = os.path.join(root_b, case_name, "combined")

        if not os.path.isdir(semantic_mask_folder):
            print(f"Skip {case_name}: mask folder not found -> {semantic_mask_folder}")
            continue

        matched += 1
        process_one_case(
            input_folder=input_folder,
            output_folder=output_folder,
            semantic_mask_folder=semantic_mask_folder,
            demo=demo
        )

    print(f"\nMatched and processed {matched} cases.")


def main():
    root_a_default = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928"
    root_b_default = "/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/model_out20260211_padding/20260630"

    root_a = sys.argv[1] if len(sys.argv) > 1 else root_a_default
    root_b = sys.argv[2] if len(sys.argv) > 2 else root_b_default

    print(f"root_a: {root_a}")
    print(f"root_b: {root_b}")

    cfg = get_cfg()
    add_swinb_config(cfg)
    cfg.merge_from_file(
        "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO_WITH_loop_focus/config.yaml"
    )
    cfg.MODEL.WEIGHTS = (
        "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/"
        "output_swin_B_3x_finetune_organoids_AO_WITH_loop_focus/model_0001049.pth"
    )
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)

    batch_process(root_a, root_b, demo)


if __name__ == "__main__":
    main()