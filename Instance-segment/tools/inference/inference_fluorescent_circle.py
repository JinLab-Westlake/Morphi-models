import cv2
import torch
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import os
from detectron2.config import get_cfg
from ...swinb.config import add_swinb_config
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

    def run_on_image(self, image,confidence_threshold = 0.0):
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

            def compute_mask_iou(mask1, mask2):
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                return intersection / union if union > 0 else 0
            
            instances = predictions["instances"].to(self.cpu_device)
            high_confidence_mask = instances.scores > confidence_threshold
            instances = instances[high_confidence_mask]

            masks = instances.pred_masks.cpu().numpy()  # (N, H, W)
            scores = instances.scores.cpu().numpy()

            keep = []

            for i in range(len(masks)):
                # if i in keep:
                #     continue
                discard = False
                for j in keep:
                    iou = compute_mask_iou(masks[i], masks[j])
                    if iou > 1:
                        if scores[i] <= scores[j]:
                            discard = True
                            break
                if not discard:
                    keep.append(i)

            # keep final instance
            instances = instances[keep]

            # Visualize
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
            
            predictions["instances"] = instances 
        

        return predictions, vis_output
    
    

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

def compute_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


cfg = get_cfg()
add_swinb_config(cfg) 

cfg.merge_from_file("/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/configs/transfiner/mask_rcnn_swinb_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/output_swin_B_3x/model_0010199.pth"

cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg.freeze()

# Initialize the tool
demo = VisualizationDemo(cfg, instance_mode=ColorMode.IMAGE)

# Input and output directories
input_folder = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/tracing/J8"
output_folder = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/tracing/J8/output_circle"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".tif", ".jpeg",".tiff")) and not filename.startswith("._"):  # Process image files only
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Read image
        image = cv2.imread(input_path)

        # Run inference
        predictions, vis_output = demo.run_on_image(image)

        
        # if there are results of instance segmentation
        if "instances" in predictions:
            instances = predictions["instances"].to(demo.cpu_device)

            
            if len(instances) == 0:
                print(f"there's no high confidence detections for {filename}")
                continue
            
            # compute the roundness of each mask
            pred_masks = instances.pred_masks.cpu().numpy()
            circularities = []
            for i, mask in enumerate(pred_masks):
                mask_uint8 = (mask.astype(np.uint8)*255)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    circ = 0.0
                else:
                    cnt = max(contours, key = cv2.contourArea)
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    circ = 4* np.pi *area / (perimeter **2) if perimeter >0 else 0.0
                circularities.append(circ)
                
            iou_threshold = 0.5
            sorted_indices = sorted(range(len(circularities)), key=lambda i: circularities[i], reverse=True)
            selected = []
            for idx in sorted_indices:
                keep_flag = True
                for kept_idx in selected:
                    iou = compute_mask_iou(pred_masks[idx], pred_masks[kept_idx])
                    if iou > iou_threshold:
                        keep_flag = False
                        break
                if keep_flag:
                    selected.append(idx)
            
            circularity_threshold = 0.2  # roundness threshold
            selected_filtered = [idx for idx in selected if circularities[idx] > circularity_threshold]

            if not selected_filtered:
                print("No instances with circularity greater than threshold.")
            else:
                # Filter the detection results based on the selected index
                filtered_instances = instances[selected_filtered]
                predictions["instances"] = filtered_instances

                print("Selected instances (by circularity):", selected_filtered)
                print("Circularity values of selected instances:", [circularities[i] for i in selected_filtered])
                
                # The filtered detection results are plotted using the Visualizer
                image_rgb = image[:, :, ::-1]
                visualizer = Visualizer(image_rgb, demo.metadata, instance_mode=ColorMode.IMAGE)
                vis_output = visualizer.draw_instance_predictions(predictions=filtered_instances)

        
        # save images
        if vis_output:
            vis_image = vis_output.get_image()[:, :, ::-1]
            cv2.imwrite(output_path, vis_image)
            print(f"Processed: {filename} -> Saved to {output_path}")
        else:
            print(f"No detections for {filename}")

print("Batch processing completed.")

