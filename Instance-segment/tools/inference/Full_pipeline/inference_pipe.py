import cv2
from detectron2.config import get_cfg
# from detectron2.utils.visualizer import ColorMode
import torch
import atexit
import bisect
import multiprocessing as mp
from collections import deque


from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import os

from detectron2.config import get_cfg

import sys
sys.path.insert(0, "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main")

import argparse
from detectron2.config import get_cfg

from swinb.config import add_swinb_config
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
                        if iou > 0.1:
                            if scores[i] <= scores[j]:
                                discard = True
                                break
                    if not discard:
                        keep.append(i)

                # 只保留非冗余实例
                instances = instances[keep]

                
                # Step 3: Remove boxes too close to image borders
                H, W = image.shape[:2]
                margin = 5  # 可调整：距离边缘小于5像素的 box 被移除

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
                
                

        return predictions, vis_output,instances
    
    

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

            


class ImageProcessor:
    def __init__(self, cfg,region_key, sytox_label, bf_image, output_folder):
        self.cfg = cfg
        self.region_key = region_key[0] 
        self.sytox_label = sytox_label
        self.bf_image = bf_image
        self.output_folder = output_folder
        self.demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def process_image(self, bf_image, region_key):
        # input_path = os.path.join(self.input_folder, filename)
        output_path = os.path.join(self.output_folder, f"{region_key[0]}+.tiff")
        
        image = cv2.imread("/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250801/20250801_A2780_NO1_P01-P04__2025-08-01T09_18_05-Measurement_1/Combined/channel_2/r01c01f01.tiff")
        
        # Run inference
        predictions, vis_output, instances = self.demo.run_on_image(image)
        
        # If no detection, skip
        if not vis_output:
            print(f"No detections for {region_key}")
            return
        
        vis_image = vis_output.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV
        cv2.imwrite(output_path, vis_image)
        print(f"Processed: {region_key} -> Saved to {output_path}")

        self.process_masks(region_key, image, self.sytox_label, instances, vis_image)
    
    def process_masks(self, filename, image, sytox_label, instances, vis_image):
        # ch2_path = os.path.join(self.input_folder.replace("channel_2", "SYTOX_labels"), filename)
        # ch2_mask = cv2.imread(ch2_path, cv2.IMREAD_GRAYSCALE)
        # if sytox_label is None:
        #     print(f"Warning: ch2 mask not found for {filename}")
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     ch2_mask = np.zeros_like(gray, dtype=np.uint8)  # img 是 ch1 图像

        output_folder = os.path.join(self.output_folder, "sytox_output")
        
        
        ch2_mask = (sytox_label > 0).astype(np.uint8)  # 二值化处理

        if len(instances) == 0 or not instances.has("pred_masks"):
            print(f"No valid instances for {filename}")
            return
        
        pred_masks = instances.pred_masks.numpy()  # shape: [N, H, W]

        # Initialize results list
        results = []
        gt_mask = np.zeros_like(ch2_mask, dtype=np.uint8)  # Initialize ground truth mask

        for i, mask in enumerate(pred_masks):
            instance_area = np.sum(mask)
            ch2_area = np.sum(ch2_mask[mask > 0])
            ratio = ch2_area / instance_area if instance_area > 0 else 0

            # Overlap region (died layer)
            overlap = (mask > 0) & (ch2_mask > 0)

            # Update ground truth mask
            gt_mask[mask > 0] = 1
            gt_mask[overlap] = 2  # Set overlap region as 2

            # Save ground truth mask
            gt_save_path = os.path.join(output_folder, filename.replace('.tiff', '_gtmask.png'))
            cv2.imwrite(gt_save_path, gt_mask)
            print(f"Saved ground truth mask to: {gt_save_path}")

            # Create overlay for visualization
            highlight_layer = np.zeros_like(vis_image, dtype=np.uint8)
            color = (255, 255, 255)  # BGR color (white)
            highlight_layer[overlap] = color

            # Write ratio on the center of each instance
            coords = np.column_stack(np.where(mask > 0))
            if coords.shape[0] == 0:
                print(f"Empty mask for instance {i} in {filename}, skipping...")
                continue
            
            center_y, center_x = coords.mean(axis=0).astype(int)
            label_text = f"{ratio:.2f}"
            cv2.putText(vis_image, label_text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            results.append({
                "Filename": filename,
                "Instance_id": i,
                "Instance_area": instance_area,
                "Died_area": ch2_area,
                "Died_over_instance": ratio * 100
            })

        # Save final image with overlay
        final_image = cv2.addWeighted(vis_image, 1.0, highlight_layer, 0.4, 0)
        final_output_path = os.path.join(output_folder, "highlight_" + filename)
        cv2.imwrite(final_output_path, final_image)
        print(f"Saved highlighted overlay to {final_output_path}")

        # Save results to Excel
        self.save_results_to_excel(results, output_folder)
    
    def save_results_to_excel(self, results, output_folder):
        excel_path = os.path.join(output_folder, "Datasets_20250514_1536well.xlsx")
        df = pd.DataFrame(results)

        # Save to Excel
        if not os.path.exists(excel_path):
            df.to_excel(excel_path, index=False)
        else:
            old_df = pd.read_excel(excel_path)
            combined_df = pd.concat([old_df, df], ignore_index=True)
            combined_df.to_excel(excel_path, index=False)

    def process_all_images(self):
        # # Process all images in the input folder
        # for filename in os.listdir(self.input_folder):
        #     if filename.endswith((".jpg", ".png", ".tif", ".jpeg", ".tiff")) and not filename.startswith('.'):
        self.process_image(self.bf_image, self.region_key)

def main(region_key,ch1_image, ch2_image, output_folder):
    cfg = get_cfg()
    add_swinb_config(cfg)  # Load swinb extension config

    # Merge from a config file
    cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_focus_finetune_died_v_organoids/config.yaml")
    cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x_focus_finetune_died_v_organoids/model_0000849.pth"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    # Initialize the ImageProcessor
    processor = ImageProcessor(cfg, region_key, ch1_image, ch2_image, output_folder)
    processor.process_all_images()
    print("Batch processing completed.")

if __name__ == "__main__":
    # Default paths
    input_folder_default = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250714_PE/20250713_A2780_24drug_1536_test__2025-07-13T11_24_07-Measurement 1/channel/ch1"
    output_folder_default = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20250714_PE/20250713_A2780_24drug_1536_test__2025-07-13T11_24_07-Measurement 1/channel/output_thresh_new"

    # Override default if arguments are passed
    input_folder = sys.argv[1] if len(sys.argv) > 1 else input_folder_default
    output_folder = sys.argv[2] if len(sys.argv) > 2 else output_folder_default

    print(f"✅ Input folder: {input_folder}")
    print(f"✅ Output folder: {output_folder}")

    main(input_folder, output_folder)