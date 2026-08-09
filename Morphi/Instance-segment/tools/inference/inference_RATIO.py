import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import os
from detectron2.config import get_cfg

from ...swinb.config import add_swinb_config
import numpy as np
import pandas as pd



class VisualizationDemo:
    def __init__(
        self,
        cfg,
        instance_mode=ColorMode.IMAGE,
        parallel=False,
    ):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0]
            if len(cfg.DATASETS.TEST)
            else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(
        self,
        image,
        confidence_threshold=0.1,
        min_mask_area=1440,
    ):
        vis_output = None
        predictions = self.predictor(image)

        image_rgb = image[:, :, ::-1]
        visualizer = Visualizer(
            image_rgb,
            self.metadata,
            instance_mode=self.instance_mode,
        )

        instances = None

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]

            vis_output = (
                visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device),
                    segments_info,
                )
            )

        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"]
                    .argmax(dim=0)
                    .to(self.cpu_device)
                )

            if "instances" in predictions:
                instances = predictions["instances"].to(
                    self.cpu_device
                )

                # 1. Confidence filtering
                if (
                    len(instances) > 0
                    and instances.has("scores")
                ):
                    score_keep = (
                        instances.scores
                        > confidence_threshold
                    )
                    instances = instances[score_keep]

                # 2. Mask area filtering
                if (
                    len(instances) > 0
                    and instances.has("pred_masks")
                ):
                    mask_areas = (
                        instances.pred_masks
                        .flatten(1)
                        .sum(dim=1)
                    )

                    area_keep = (
                        mask_areas >= min_mask_area
                    )

                    instances = instances[area_keep]

                # 3. Mask IoU filtering
                if (
                    len(instances) > 0
                    and instances.has("pred_masks")
                ):
                    def compute_mask_iou(mask1, mask2):
                        intersection = (
                            mask1 & mask2
                        ).sum().item()

                        union = (
                            mask1 | mask2
                        ).sum().item()

                        return (
                            intersection / union
                            if union > 0
                            else 0
                        )

                    masks = instances.pred_masks
                    scores = (
                        instances.scores.numpy()
                    )

                    keep = []

                    for i in range(len(masks)):
                        discard = False

                        for j in keep:
                            iou = compute_mask_iou(
                                masks[i],
                                masks[j],
                            )

                            if (
                                iou > 0.1
                                and scores[i] <= scores[j]
                            ):
                                discard = True
                                break

                        if not discard:
                            keep.append(i)

                    instances = instances[keep]

                # 4. Remove instances touching image boundary
                if (
                    len(instances) > 0
                    and instances.has("pred_boxes")
                ):
                    height, width = image.shape[:2]
                    margin = 5

                    boxes = (
                        instances.pred_boxes.tensor
                        .numpy()
                    )

                    final_keep = []

                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box

                        if (
                            x1 > margin
                            and y1 > margin
                            and x2 < width - margin
                            and y2 < height - margin
                        ):
                            final_keep.append(i)

                    instances = instances[final_keep]

                vis_output = (
                    visualizer.draw_instance_predictions(
                        predictions=instances
                    )
                )

        return predictions, vis_output, instances




cfg = get_cfg()
add_swinb_config(cfg)  # Load the swinb extension configuration

cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO_WITH_loop_focus/config.yaml")
cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_finetune_organoids_AO_WITH_loop_focus/model_0001049.pth"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg.freeze()

# Initialize the tool
demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)

import sys

input_folder_default = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250714_PE/20250713-YOKOWGAWA-A2780-24drugs-1536-test_20250713_121140/channel/ch1"
output_folder_default = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250714_PE/20250713-YOKOWGAWA-A2780-24drugs-1536-test_20250713_121140/channel/output"

input_folder = sys.argv[1] if len(sys.argv) > 1 else input_folder_default
output_folder = sys.argv[2] if len(sys.argv) > 2 else output_folder_default

print(f"✅ Input folder: {input_folder}")
print(f"✅ Output folder: {output_folder}")

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
        

        # Save the result if detection exists
        if vis_output:
            vis_image = vis_output.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV
            cv2.imwrite(output_path, vis_image)
            print(f"Processed: {filename} -> Saved to {output_path}")

            # Read the corresponding ch2 image
            ch2_path = os.path.join(input_folder.replace("ch1", "SYTOX_labels"), filename)
            ch2_mask = cv2.imread(ch2_path, cv2.IMREAD_GRAYSCALE)
            if ch2_mask is None:
                print(f"Warning: ch2 mask not found for {filename}")
                gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # continue
                ch2_mask = np.zeros_like(gray, dtype=np.uint8)  # img is the ch1 image
                
            ch2_mask = (ch2_mask > 0).astype(np.uint8)  # binary

            # instances = predictions["instances"].to("cpu")
            print(len(predictions))
            print(len(instances))
            if len(instances) == 0 or not instances.has("pred_masks"):
                print(f"No valid instances for {filename}")
                continue

            pred_masks = instances.pred_masks.numpy()  # shape: [N, H, W]

            results = []
            
            
            """
            ground truth 
            """
            # ground truth mask：0 -> bg
            gt_mask = np.zeros_like(ch2_mask, dtype=np.uint8)

            for i, mask in enumerate(pred_masks):
                instance_area = np.sum(mask)
                ch2_area = np.sum(ch2_mask[mask > 0])
                ratio = ch2_area / instance_area if instance_area > 0 else 0

                # died layer overlaped
                overlap = (mask > 0) & (ch2_mask > 0)

                # Start by setting the pred_mask area to 1
                gt_mask[mask > 0] = 1
                
                # Set the overlap to 2 (overwrites the original 1).
                gt_mask[overlap] = 2


            # saving ground truth mask
            gt_save_path = os.path.join(output_folder, filename.replace('.tiff', '_gtmask.png'))
            cv2.imwrite(gt_save_path, gt_mask)
            print(f"Saved ground truth mask to: {gt_save_path}")

            """
            Show the prediction results, on the original plot
            """
            # Create a layer for the transparent overlay
            highlight_layer = np.zeros_like(vis_image, dtype=np.uint8)
            
            for i, mask in enumerate(pred_masks):
                instance_area = np.sum(mask)  # instance area
                ch2_area = np.sum(ch2_mask[mask > 0])  # The number of valid pixels in ch2 within that instance
                ratio = ch2_area / instance_area if instance_area > 0 else 0
                
                # overlap the part of the instance where the died layer has sign
                overlap = (mask > 0) & (ch2_mask > 0)
                color = (255, 255, 255)  # BGR
                for c in range(3):
                    highlight_layer[..., c][overlap] = color[c]
                
                # Write the scale value in the center of the instance
                coords = np.column_stack(np.where(mask > 0))
                if coords.shape[0] == 0:
                    print(f"Empty mask for instance {i} in {filename}, skipping...")
                    continue
                
                center_y, center_x = coords.mean(axis=0).astype(int)
                label_text = f"{ratio:.2f}"
                
                if vis_image.dtype != np.uint8:
                    vis_image = (vis_image * 255).astype(np.uint8)
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                
                cv2.putText(vis_image, label_text, (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
                vis_image = vis_image[:, :, ::-1]
                
                results.append({
                    "Filename": filename,
                    "Instance_id": i,
                    "Instance_area": instance_area,
                    "Died_area": ch2_area,
                    "Died_over_instance": ratio*100
                })

            # Saving statistics
            excel_path = os.path.join(output_folder, "Datasets_YOKOWGAWA_20250714.xlsx")
            df = pd.DataFrame(results)

            # Save as excel
            if not os.path.exists(excel_path):
                df.to_excel(excel_path, index=False)
            else:
                old_df = pd.read_excel(excel_path)
                combined_df = pd.concat([old_df, df], ignore_index=True)
                combined_df.to_excel(excel_path, index=False)

            final_image = cv2.addWeighted(vis_image, 1.0, highlight_layer, 0.4, 0)
            final_output_path = os.path.join(output_folder, "highlight_" + filename)
            cv2.imwrite(final_output_path, final_image)
            print(f"Saved highlighted overlay to {final_output_path}")
            
        else:
            print(f"No detections for {filename}")

print("Batch processing completed.")