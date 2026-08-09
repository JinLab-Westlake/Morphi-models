import numpy as np
from sklearn.metrics import jaccard_score,precision_score,recall_score,f1_score
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import cv2
from PIL import Image




class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
    
    
class MultiTaskSegmentationHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # Main segmentation output: Segmented cell instance region (binary or multi-class)
        self.segmentation_head = nn.Conv2d(in_channels, 1, kernel_size=1)

        # live organoid mask output (0~1)
        self.live_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # dead organoid mask output (0~1)
        self.dead_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Probability of death output
        self.death_prob_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Or without sigmoid, train with MSE
        )

    def forward(self, x):
        seg_mask = self.segmentation_head(x)  # raw logits, with BCEWithLogitsLoss
        live_mask = self.live_head(x)         # Probability map of living cells
        dead_mask = self.dead_head(x)         # Dead cell probability plot
        death_prob = self.death_prob_head(x)  # Plot of continuous probability of death

        return {
            "segmentation": seg_mask,
            "live": live_mask,
            "dead": dead_mask,
            "death_prob": death_prob
        }





def calculate_metric_percase(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        
        # calculate mIOU
        mIOU = jaccard_score(gt.flatten(),pred.flatten())
        
        # calculate precision, recall and f1-score
        recall = recall_score(gt.flatten(),pred.flatten())
        precision = precision_score(gt.flatten(),pred.flatten())
        f1 = f1_score(gt.flatten(),pred.flatten())
        
        return dice, hd95,mIOU,recall,precision,f1
    
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0,0,0,1,0
    else:
        return 0, 0,0,0,0,0






def save_as_original_png(image, path, case, suffix):
    """
    Helper function to save the original grayscale image.
    """
    img = Image.fromarray(image.astype(np.uint8), mode='L')  # Keep the original gray value
    img.save(f"{path}/{case}_{suffix}.png")

def save_as_gray_png(image, path, case, suffix, num_classes):
    """
    param:
    case: The name of imgae 
    suffix: the type of image(like, image/prediction/groundtruth...)
    """
    max_value = 255
    # The step size of each category is calculated so that the category values are distributed in the range 0-255
    step = max_value // (num_classes - 1)
    contrasted_image = (image * step).astype(np.uint8)
    
    img = Image.fromarray(contrasted_image, mode='L')  # gray mode
    img.save(f"{path}/{case}_{suffix}.png")




def draw_and_save_boxes(image, target_boundaries, save_path, case, suffix):
    """
    Label boundary of object and save image.
    Args:
        image (numpy.ndarray): Original image (H,W) or (H,W,C).
        target_boundaries (list): List of boundaries, each element is a dictionary.
        save_path (str): Directory where image will be saved.
        case (str): Case identifier for naming.
        suffix (str): Suffix to add to the filename.
    """
    # If image is single channel, convert to BGR for color visualization
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    # Set color and thickness for the boxes
    box_color = (0, 255, 0)  # Green
    box_thickness = 1
    
    # 检查 target_boundaries 的内容和格式
    # print("target_boundaries:", target_boundaries)
    # for boundary in target_boundaries:
    #     print(type(boundary), boundary)  # 打印每个边界的类型和内容
    
    # Draw each boundary box on the image
    for boundary in target_boundaries:
        top = boundary['top']
        left = boundary['left']
        right = boundary['right']
        down = boundary['down']
        
        # Draw rectangle
        cv2.rectangle(image, (left, top), (right, down), box_color, box_thickness)
        
        # Label each object with its index
        text = f"ID:{boundary['target_index']}"
        cv2.putText(image, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
    
    # Ensure save_path is a directory, add file name with case and suffix
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    output_file = os.path.join(save_path, f"{case}_{suffix}.png")
    
    # Save the image with the box boundaries
    cv2.imwrite(output_file, image)




# 改resize为填充

import numpy as np
from skimage.transform import resize

import numpy as np
from skimage.transform import resize

def resize_and_pad_to_patch(image, patch_size, pad_mode='constant', pad_val=0):
    """
    图像根据 patch size 调整：
    - 如果最长边 ≤ patch_size，保持原比例直接 padding；
    - 如果最长边 > patch_size，先缩放最长边到 patch_size，再 padding。

    Args:
        image: 2D numpy array, 原图
        patch_size: (H, W)，目标尺寸
        pad_mode: np.pad 的模式（默认 constant）
        pad_val: 填充值（默认 0）

    Returns:
        resized_padded_image: 最终尺寸为 patch_size 的图像
    """
    orig_h, orig_w = image.shape
    target_h, target_w = patch_size

    # 判断是否需要缩放
    if orig_h <= target_h and orig_w <= target_w:
        # 直接 padding 到目标大小
        pad_h = target_h - orig_h
        pad_w = target_w - orig_w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        padded = np.pad(image, 
                        ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode=pad_mode, 
                        constant_values=pad_val)
        return padded
    
    else:
        # 需要先缩放，再 padding
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        scale = min(scale_h, scale_w)

        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        resized = resize(image, (new_h, new_w), order=3, mode='reflect', anti_aliasing=True)

        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        padded = np.pad(resized, 
                        ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode=pad_mode, 
                        constant_values=pad_val)
        return padded





import pandas as pd
from scipy.ndimage import label, find_objects
import torch
from scipy.ndimage import zoom
import os

def inference_single_volum_whole(image, ground_truth, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    # 将图像和标签转换为 NumPy 数组，并确保是二维的
    image = image.squeeze(0).squeeze(0).cpu().detach().numpy()
    # label = label.squeeze(0).cpu().detach().numpy()

    # 获取图像的大小
    x, y = image.shape[0], image.shape[1]

    # 如果图像大小与 patch_size 不匹配，调整图像大小
    if x != patch_size[0] or y != patch_size[1]:
        # 确保输入的 zoom 比例与图像维度一致
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        # image = resize_and_pad_to_patch(image, patch_size)
        print(image.shape)

    # 将图像转换为 Tensor
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()

    # 推理
    with torch.no_grad():
        # outputs = net(input_tensor)
        # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        
        # outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        outputs = net(input_tensor)
        # 如果输出是 tuple，例如 (logits, aux), 只取 logits
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)

        # out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)

        
        prediction = out.cpu().detach().numpy()

        # 如果图像进行了缩放，则恢复到原始大小
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    # record the edge box of each object
    target_boundaries = []
    
    # 保存用于Excel的数据
    excel_data = []
    
    # record the object area (connective area)
    labeled_array, num_features = label(prediction)
    objects = find_objects(labeled_array)
    
    for i, obj in enumerate(objects, start=1):
        if obj is not None:
            # get the edge of each object
            top, down = obj[0].start, obj[0].stop - 1
            left, right = obj[1].start, obj[1].stop - 1
            
            #---------------------
            # what if the size if too small to be a noise; then need to be removed
            if (down-top)*(right-left) > 32*32:
            
                # 缩放边界框坐标到 patch_size
                scale_x = patch_size[0] / x
                scale_y = patch_size[1] / y
                top_scaled, down_scaled = int(top * scale_x), int(down * scale_x)
                left_scaled, right_scaled = int(left * scale_y), int(right * scale_y)
                
                # 计算中心点坐标
                central_x = (left_scaled + right_scaled) // 2
                central_y = (top_scaled + down_scaled) // 2

                # 记录目标边界和中心点信息
                target_boundaries.append({
                    'target_index': i,
                    'top': top_scaled,
                    'left': left_scaled,
                    'right': right_scaled,
                    'down': down_scaled,
                    'central_x': central_x,
                    'central_y': central_y
                })

                # 添加到Excel数据
                excel_data.append({
                    'imagename': case,
                    'target_index': i,
                    'top': top,
                    'left': left,
                    'right': right,
                    'down': down,
                    'central_x': (left + right) // 2,
                    'central_y': (top + down) // 2
                })
            
            else:
                continue
            
            
    # 如果需要保存推理结果
    if test_save_path is not None:
        save_as_original_png(image, test_save_path, case, "img")
        save_as_gray_png(prediction, test_save_path, case, "pred", 3)
    
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)  # 确保保存目录存在
        excel_save_path = os.path.join(test_save_path, f"{case}_results.xlsx")
        
        # 将推理的图像和边界框结果保存
        draw_and_save_boxes(image, target_boundaries, test_save_path, case, 'labeled_box')


        
        # save result to excel
        df = pd.DataFrame(excel_data)
        df.to_excel(excel_save_path, index=False)
        print(f"Results saved to {excel_save_path}")

    return prediction, target_boundaries




def inference_single_volum_patches(image, ground_truth, net, classes, patch_size=[256, 256], test_save_path=None, case=None):
    # Convert the images and labels to NumPy arrays and make sure they are two-dimensional
    image = image.squeeze(0).squeeze(0).cpu().detach().numpy()

    # Gets the size of the image
    x, y = image.shape[0], image.shape[1]

    # If the image size does not match the patch_size, resize the image
    if x != patch_size[0] or y != patch_size[1]:
        # Make sure the input zoom scale is consistent with the image dimensions
        image = resize_and_pad_to_patch(image, patch_size)

    # transfer image to Tensor
    input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()

    # inference
    with torch.no_grad():
        outputs = net(input_tensor)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()

        # If the image is scaled, it is restored to its original size
        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    # record the edge box of each object
    target_boundaries = []
    
    # Save the data for Excel
    excel_data = []
    
    # record the object area (connective area)
    labeled_array, num_features = label(prediction)
    objects = find_objects(labeled_array)
    
    for i, obj in enumerate(objects, start=1):
        if obj is not None:
            # get the edge of each object
            top, down = obj[0].start, obj[0].stop - 1
            left, right = obj[1].start, obj[1].stop - 1
            
            # Scale the bounding box coordinates to patch_size
            scale_x = patch_size[0] / x
            scale_y = patch_size[1] / y
            top_scaled, down_scaled = int(top * scale_x), int(down * scale_x)
            left_scaled, right_scaled = int(left * scale_y), int(right * scale_y)
            
            # Calculate the center point coordinates
            central_x = (left_scaled + right_scaled) // 2
            central_y = (top_scaled + down_scaled) // 2

            # The object boundary and center point information were recorded
            target_boundaries.append({
                'target_index': i,
                'top': top_scaled,
                'left': left_scaled,
                'right': right_scaled,
                'down': down_scaled,
                'central_x': central_x,
                'central_y': central_y
            })

    
    # Save the inference results if needed
    if test_save_path is not None:
        save_as_original_png(image, test_save_path, case, "img")
        save_as_gray_png(prediction, test_save_path, case, "pred", 3)






def test_single_volume(
    image,
    label,
    net,
    classes,
    patch_size=[256, 256],
    test_save_path=None,
    case=None,
    z_spacing=1
):
    """
    image: [1, 1, H, W]
    label: [1, H, W]
    class index:
        0 = background
        1 = live
        2 = dead
    return:
        {
            "prediction": HxW numpy array,
            "class_metrics": {
                1: [dice, hd95, iou, recall, precision, f1],
                2: [dice, hd95, iou, recall, precision, f1]
            },
            "fg_metrics": [dice, hd95, iou, recall, precision, f1],
            "gt_dead_ratio": float or nan,
            "pred_dead_ratio": float or nan,
        }
    """
    image_np = image.squeeze(0).squeeze(0).cpu().detach().numpy()
    label_np = label.squeeze(0).cpu().detach().numpy()

    x, y = image_np.shape

    if x != patch_size[0] or y != patch_size[1]:
        image_resized = zoom(image_np, (patch_size[0] / x, patch_size[1] / y), order=3)
    else:
        image_resized = image_np

    input_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()

    with torch.no_grad():
        outputs = net(input_tensor)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()

        if x != patch_size[0] or y != patch_size[1]:
            prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)

    class_metrics = {}
    for i in range(1, classes):
        pred_i = (prediction == i).astype(np.bool_)
        label_i = (label_np == i).astype(np.bool_)

        if pred_i.shape != label_i.shape:
            print(f"Warning: Shape mismatch between prediction and label for class {i}")
            continue

        class_metrics[i] = calculate_metric_percase(pred_i, label_i)

    # foreground = live + dead
    pred_fg = (prediction > 0).astype(np.bool_)
    label_fg = (label_np > 0).astype(np.bool_)
    fg_metrics = calculate_metric_percase(pred_fg, label_fg)

    # dead ratio
    pred_live = np.sum(prediction == 1)
    pred_dead = np.sum(prediction == 2)
    gt_live = np.sum(label_np == 1)
    gt_dead = np.sum(label_np == 2)

    pred_fg_area = pred_live + pred_dead
    gt_fg_area = gt_live + gt_dead

    pred_dead_ratio = np.nan if pred_fg_area == 0 else pred_dead / pred_fg_area
    gt_dead_ratio = np.nan if gt_fg_area == 0 else gt_dead / gt_fg_area

    if test_save_path is not None:
        save_as_original_png(image_np, test_save_path, case, "img")
        save_as_gray_png(prediction, test_save_path, case, "pred", 3)

    return {
        "prediction": prediction,
        "class_metrics": class_metrics,
        "fg_metrics": fg_metrics,
        "gt_dead_ratio": gt_dead_ratio,
        "pred_dead_ratio": pred_dead_ratio,
    }