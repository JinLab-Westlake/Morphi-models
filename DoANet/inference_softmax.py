import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# from datasets.dataset_synapse import Synapse_dataset
from datasets.cell_datasets import Synapse_dataset
from utils import inference_single_volum_patches
from networks.DoANet import DoANet
from trainer import trainer_synapse
from config import get_config
from PIL import Image

from utils import resize_and_pad_to_patch 
from pathlib import Path




from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch


"""
original version
"""
# class ResizePadGenerator:
#     def __init__(self, output_size=(224, 224)):
#         self.target_size = output_size[0]

#     def __call__(self, image_np):
#         # 输入 image_np 是 numpy 数组（灰度）
#         image = Image.fromarray(image_np).convert('L')

#         iw, ih = image.size
#         if iw > self.target_size or ih > self.target_size:
#             scale = self.target_size / max(iw, ih)
#             new_w, new_h = int(iw * scale), int(ih * scale)
#             image = image.resize((new_w, new_h), resample=Image.BILINEAR)
#         else:
#             new_w, new_h = iw, ih

#         pad_left   = (self.target_size - new_w) // 2
#         pad_top    = (self.target_size - new_h) // 2
#         pad_right  = self.target_size - new_w - pad_left
#         pad_bottom = self.target_size - new_h - pad_top

#         image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
#         image = TF.to_tensor(image)  # 转为 [1, H, W] 的 Tensor，值域 0~1

#         return image



"""
for z-score
"""
# class ResizePadGenerator:
#     def __init__(self, output_size=(224, 224)):
#         self.target_size = output_size[0]

#     def __call__(self, image_np):
#         # 输入 image_np 是 numpy 数组 (灰度 0~255)
#         image = Image.fromarray(image_np).convert('L')

#         # 等比缩放
#         iw, ih = image.size
#         scale = self.target_size / max(iw, ih)
#         new_w, new_h = int(iw * scale), int(ih * scale)
#         image = image.resize((new_w, new_h), resample=Image.BILINEAR)

#         # Padding
#         pad_left   = (self.target_size - new_w) // 2
#         pad_top    = (self.target_size - new_h) // 2
#         pad_right  = self.target_size - new_w - pad_left
#         pad_bottom = self.target_size - new_h - pad_top

#         image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        
#         # Z-score 标准化 (必须与训练逻辑完全一致)
#         img_np = np.array(image).astype(np.float32) / 255.0
#         mean, std = img_np.mean(), img_np.std()
#         img_np = (img_np - mean) / (std + 1e-8)  # 拉开对比度

#         # 转为 Tensor
#         image_tensor = torch.from_numpy(img_np).unsqueeze(0).float() # [1, H, W]

#         pad_info = {
#             "top": pad_top, "bottom": pad_bottom,
#             "left": pad_left, "right": pad_right,
#             "resized_w": new_w, "resized_h": new_h
#         }
        
#         return image_tensor, pad_info


# class ResizePadGenerator:
#     def __init__(self, output_size=(224, 224)):
#         self.target_size = output_size[0]  # 目标尺寸的宽度（或高度，取决于应用场景）

#     def __call__(self, image_np):
#         # 输入 image_np 是 numpy 数组 (灰度图，0~255)
#         image = Image.fromarray(image_np).convert('L')

#         # 记录原始图像尺寸
#         original_w, original_h = image.size

#         # 等比缩放
#         iw, ih = image.size
#         scale = self.target_size / max(iw, ih)
#         new_w, new_h = int(iw * scale), int(ih * scale)
#         image = image.resize((new_w, new_h), resample=Image.BILINEAR)

#         # Padding
#         pad_left   = (self.target_size - new_w) // 2
#         pad_top    = (self.target_size - new_h) // 2
#         pad_right  = self.target_size - new_w - pad_left
#         pad_bottom = self.target_size - new_h - pad_top

#         image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        
#         # Z-score 标准化 (必须与训练时使用的标准化一致)
#         img_np = np.array(image).astype(np.float32) / 255.0
#         mean, std = img_np.mean(), img_np.std()
#         img_np = (img_np - mean) / (std + 1e-8)  # 标准化

#         # 转换为 Tensor
#         image_tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # [1, H, W]

#         # 包含原始尺寸信息
#         pad_info = {
#             "top": pad_top, "bottom": pad_bottom,
#             "left": pad_left, "right": pad_right,
#             "resized_w": new_w, "resized_h": new_h,
#             "original_w": original_w, "original_h": original_h  # 保存原始尺寸
#         }
        
#         return image_tensor, pad_info




"""
for z-score & size-adjusted
"""
# class ResizePadGenerator:
#     def __init__(self, output_size=(224, 224)):
#         self.target_size = output_size[0]  # 目标尺寸的宽度（或高度，取决于应用场景）

#     def __call__(self, image_np):
#         # 输入 image_np 是 numpy 数组 (灰度图，0~255)
#         image = Image.fromarray(image_np).convert('L')
        
#         original_w, original_h = image.size

#         iw, ih = image.size

#         # 只有当图像尺寸大于目标尺寸时，才进行缩放
#         if iw > self.target_size or ih > self.target_size:
#             scale = self.target_size / max(iw, ih)  # 等比缩放
#             new_w, new_h = int(iw * scale), int(ih * scale)
#             image = image.resize((new_w, new_h), resample=Image.BILINEAR)
#         else:
#             new_w, new_h = iw, ih  # 保持原始大小，不进行缩放

#         # 计算填充量
#         pad_left   = (self.target_size - new_w) // 2
#         pad_top    = (self.target_size - new_h) // 2
#         pad_right  = self.target_size - new_w - pad_left
#         pad_bottom = self.target_size - new_h - pad_top

#         # 对图像进行填充
#         image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        
#         # Z-score 标准化 (必须与训练时使用的标准化一致)
#         img_np = np.array(image).astype(np.float32) / 255.0
#         mean, std = img_np.mean(), img_np.std()
#         img_np = (img_np - mean) / (std + 1e-8)  # 标准化

#         # 转换为 Tensor
#         image_tensor = torch.from_numpy(img_np).unsqueeze(0).float()  # [1, H, W]

#         pad_info = {
#             "top": pad_top, "bottom": pad_bottom,
#             "left": pad_left, "right": pad_right,
#             "resized_w": new_w, "resized_h": new_h,
#             "original_w": original_w, "original_h": original_h
#         }
        
#         return image_tensor, pad_info





"""
without z-score
"""
class ResizePadGenerator:
    def __init__(self, output_size=(224, 224)):
        self.target_size = output_size[0]

    def __call__(self, image_np):
        # 输入 image_np 是 numpy 数组（灰度）
        image = Image.fromarray(image_np).convert('L')
        
        original_w, original_h = image.size

        iw, ih = image.size
        # if iw > self.target_size or ih > self.target_size:
        scale = self.target_size / max(iw, ih)
        new_w, new_h = int(iw * scale), int(ih * scale)
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        # else:
        #     new_w, new_h = iw, ih

        pad_left   = (self.target_size - new_w) // 2
        pad_top    = (self.target_size - new_h) // 2
        pad_right  = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        image = TF.to_tensor(image)  # [1, H, W], 0~1

        pad_info = {
            "top": pad_top, "bottom": pad_bottom,
            "left": pad_left, "right": pad_right,
            "resized_w": new_w, "resized_h": new_h,
            "original_w": original_w, "original_h": original_h
        }
        # 返回(图像Tensor, padding信息)
        return image, pad_info





# define png datasets

# class PNGDataset(torch.utils.data.Dataset):
#     def __init__(self, image_dir, label_dir, list_dir):
#         self.image_dir = image_dir
#         self.label_dir = label_dir 
#         # load list of image
#         with open(list_dir, 'r') as f:
#             self.image_list = [line.strip() for line in f]
    
#     def __len__(self):
#         return len(self.image_list)
    
#     def __getitem__(self,idx):
#         case_name = self.image_list[idx]
#         image_path = os.path.join(self.image_dir, case_name)
#         # label_path = os.path.join(self.label_dir, 'label_'+case_name)
#         label_path = os.path.join(self.label_dir, case_name)
        
        
#         # transfer the image into gray scale image
#         image = np.array(Image.open(image_path).convert('L'))
#         label = np.array(Image.open(label_path).convert('L'))
        
#         # expand image into the shape of input (batch_size,channels,height,weight)
#         image = torch.from_numpy(image).unsqueeze(0).float()
#         label = torch.from_numpy(label).long()
        
#         return {"image":image, "label":label, "case_name":case_name}

# class PNGDataset(torch.utils.data.Dataset):
#     def __init__(self, image_dir, list_dir):
#         self.image_dir = image_dir
#         # load list of images
#         with open(list_dir, 'r') as f:
#             self.image_list = [line.strip() for line in f]
    
#     def __len__(self):
#         return len(self.image_list)
    
#     def __getitem__(self, idx):
#         case_name = self.image_list[idx]
#         image_path = os.path.join(self.image_dir, case_name)
        
#         # Load image and convert to grayscale
#         image = np.array(Image.open(image_path).convert('L'))
        
#         # Expand image to match input shape (batch_size, channels, height, width)
#         image = torch.from_numpy(image).unsqueeze(0).float()
        
#         return {"image": image, "case_name": case_name}


# class PNGDataset(torch.utils.data.Dataset):
#     def __init__(self, image_dir, list_dir, transform=None):
#         self.image_dir = image_dir
#         with open(list_dir, 'r') as f:
#             self.image_list = [line.strip() for line in f]
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         case_name = self.image_list[idx]
#         image_path = os.path.join(self.image_dir, case_name)

#         # Load image as grayscale
#         image = np.array(Image.open(image_path).convert('L'))

#         if self.transform:
#             image = self.transform(image)  # Apply resize, pad, normalize

#         # Return with case name
#         return {"image": image, "case_name": case_name}

class PNGDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, list_dir, transform=None):
        self.image_dir = image_dir
        with open(list_dir, 'r') as f:
            self.image_list = [line.strip() for line in f]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        case_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, case_name)

        image = np.array(Image.open(image_path).convert('L'))

        pad_info = None
        if self.transform:
            out = self.transform(image)
            if isinstance(out, tuple):
                image, pad_info = out
            else:
                image = out

        return {"image": image, "case_name": case_name, "pad_info": pad_info}


def unpad_tensor_chw(t: torch.Tensor, pad_info: dict) -> torch.Tensor:
    """
    输入: t [C,H,W] (Tensor)，输出: 去padding的 [C,h',w']。
    """
    top, bottom = pad_info["top"], pad_info["bottom"]
    left, right = pad_info["left"], pad_info["right"]
    _, H, W = t.shape
    h0, h1 = top, H - bottom
    w0, w1 = left, W - right
    return t[:, h0:h1, w0:w1]





# def inference(args, model, test_save_path=None):
#     transform = ResizePadGenerator(output_size=(args.img_size, args.img_size))
#     db_test = PNGDataset(image_dir=args.volume_path, list_dir=args.list_dir, transform=transform)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()

#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         image, case_name = sampled_batch["image"].cuda(), sampled_batch["case_name"][0]  # image: [1, 1, H, W]

#         with torch.no_grad():
#             logits, _ = model(image)
#             probs = torch.softmax(logits, dim=1)  # [1, C, H, W]

#         # 保存输入图像
#         input_img = (image[0, 0].cpu().numpy() * 255).astype(np.uint8)
#         Image.fromarray(input_img).save(os.path.join(test_save_path, f"{case_name}_input.png"))

#         # 保存每个类的概率图
#         for cls_idx in range(probs.shape[1]):
#             prob_map = probs[0, cls_idx].cpu().numpy()
#             prob_map = prob_map / prob_map.max() if prob_map.max() > 0 else prob_map
#             Image.fromarray((prob_map * 255).astype(np.uint8)).save(
#                 os.path.join(test_save_path, f"{case_name}_prob_class{cls_idx}.png")
#             )

#         # 保存 argmax 预测图
#         pred_label = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
#         Image.fromarray((pred_label * (255 // max(1, probs.shape[1]-1))).astype(np.uint8)).save(
#             os.path.join(test_save_path, f"{case_name}_pred.png")
#         )

#         logging.info(f'Inference for case {case_name} completed')

#     return "Inference Finished!"



# def inference(args, model, test_save_path=None):
#     transform = ResizePadGenerator(output_size=(args.img_size, args.img_size))
#     db_test = PNGDataset(image_dir=args.volume_path, list_dir=args.list_dir, transform=transform)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()

#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         image = sampled_batch["image"].cuda()             # [1, 1, H, W] (已pad)
#         case_name = sampled_batch["case_name"][0]
#         pad_info = sampled_batch["pad_info"][0] if isinstance(sampled_batch["pad_info"], list) else sampled_batch["pad_info"]
#         # pad_info 可能是 None（如果无transform），这里建议断言
#         assert pad_info is not None, "pad_info 缺失，请确认 transform 返回 (image, pad_info)"

#         with torch.no_grad():
#             logits, _ = model(image)                     # [1, C, H, W]
#             probs = torch.softmax(logits, dim=1)         # [1, C, H, W]

#         # 1) 保存去padding后的输入图像
#         # image: [1,1,H,W] -> [1,H,W]
#         inp_chw = image[0]                                # [1,H,W]
#         inp_unpad = unpad_tensor_chw(inp_chw, pad_info)   # [1,h',w']
#         input_img_np = (inp_unpad[0].cpu().numpy() * 255).astype(np.uint8)
#         if test_save_path:
#             Image.fromarray(input_img_np).save(os.path.join(test_save_path, f"{case_name}_input_unpad.png"))

#         # 2) 保存去padding后的每类概率图
#         # probs: [1,C,H,W] -> [C,H,W]
#         probs_chw = probs[0].cpu()                        # [C,H,W]
#         probs_unpad = unpad_tensor_chw(probs_chw, pad_info)   # [C,h',w']

#         if test_save_path:
#             C = probs_unpad.shape[0]
#             for cls_idx in range(C):
#                 pm = probs_unpad[cls_idx].numpy()
#                 pm = pm / pm.max() if pm.max() > 0 else pm
#                 Image.fromarray((pm * 255).astype(np.uint8)).save(
#                     os.path.join(test_save_path, f"{case_name}_prob_class{cls_idx}_unpad.png")
#                 )

#         # 3) 保存去padding后的 argmax 预测图
#         pred_label = torch.argmax(probs[0], dim=0)        # [H,W] (仍有pad)
#         pred_unpad = pred_label[pad_info["top"]:pred_label.shape[0]-pad_info["bottom"],
#                                  pad_info["left"]:pred_label.shape[1]-pad_info["right"]]  # [h',w']
#         pred_np = pred_unpad.cpu().numpy()
#         if test_save_path:
#             # 映射到可视化灰度（简单线性映射；也可直接保存 label PNG）
#             vis = (pred_np * (255 // max(1, probs.shape[1]-1))).astype(np.uint8)
#             Image.fromarray(vis).save(os.path.join(test_save_path, f"{case_name}_pred_unpad.png"))

#         logging.info(f'Inference for case {case_name} completed (saved unpadded)')
#     return "Inference Finished!"



def inference(args, model, test_save_path=None):
    transform = ResizePadGenerator(output_size=(args.img_size, args.img_size))
    db_test = PNGDataset(image_dir=args.volume_path, list_dir=args.list_dir, transform=transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image = sampled_batch["image"].cuda()             # [1, 1, H, W] (已pad)
        case_name = sampled_batch["case_name"][0]
        case_name = os.path.splitext(os.path.basename(case_name))[0]
        # print(f"casename{case_name}")
        pad_info = sampled_batch["pad_info"][0] if isinstance(sampled_batch["pad_info"], list) else sampled_batch["pad_info"]
        
        # Assert pad_info is not None
        assert pad_info is not None, "pad_info 缺失，请确认 transform 返回 (image, pad_info)"

        # with torch.no_grad():
        #     logits, _ = model(image)                     # [1, C, H, W]
        #     probs = torch.softmax(logits, dim=1)         # [1, C, H, W]
        with torch.no_grad():
            # (death_prob_logits 或 death_prob_map)
            logits, death_prob = model(image)  # [1, C, H, W] 和 [1, 1, H, W]
            probs = torch.softmax(logits, dim=1)
            
            # # 如果 death_prob 没有经过 sigmoid，这里补充一下；如果模型内部已经做过可省略
            # death_prob = torch.sigmoid(death_prob) if death_prob.shape[1] == 1 else death_prob

        # 1) 保存去padding后的输入图像
        inp_chw = image[0]  # [1, H, W]
        inp_unpad = unpad_tensor_chw(inp_chw, pad_info)  # [1, h', w']
        input_img_np = (inp_unpad[0].cpu().numpy() * 255).astype(np.uint8)
        
        if test_save_path:
            # 通过 pad_info 映射回原始图像尺寸
            original_w, original_h = pad_info["original_w"], pad_info["original_h"]
            input_img_resized = Image.fromarray(input_img_np).resize((original_w, original_h), Image.NEAREST)
            # path = os.path.join(test_save_path, "input",f"{case_name}_input_unpad.png")
            # print(f"save_path: {path}")
            input_dir = os.path.join(test_save_path, "input")
            os.makedirs(input_dir, exist_ok=True)
            input_img_resized.save(os.path.join(input_dir,f"{case_name}_input_unpad.png"))
            

        # 2) 保存去padding后的每类概率图
        probs_chw = probs[0].cpu()  # [C, H, W]
        probs_unpad = unpad_tensor_chw(probs_chw, pad_info)  # [C, h', w']
        
        if test_save_path:
            C = probs_unpad.shape[0]
            for cls_idx in range(C):
                pm = probs_unpad[cls_idx].numpy()
                pm = pm / pm.max() if pm.max() > 0 else pm
                # 映射回原始图像尺寸并保存
                pm_resized = Image.fromarray((pm * 255).astype(np.uint8)).resize((original_w, original_h), Image.NEAREST)
                prob_dir = os.path.join(test_save_path, "prob")
                os.makedirs(prob_dir, exist_ok=True)
                pm_resized.save(os.path.join(prob_dir, f"{case_name}_prob_class{cls_idx}_unpad.png"))

        # 3) 保存去padding后的 argmax 预测图
        pred_label = torch.argmax(probs[0], dim=0)  # [H, W] (仍有pad)
        pred_unpad = pred_label[pad_info["top"]:pred_label.shape[0] - pad_info["bottom"],
                                 pad_info["left"]:pred_label.shape[1] - pad_info["right"]]  # [h', w']
        pred_np = pred_unpad.cpu().numpy()
        
        if test_save_path:
            # 映射回原始图像尺寸并保存
            vis = (pred_np * (255 // max(1, probs.shape[1] - 1))).astype(np.uint8)
            vis_resized = Image.fromarray(vis).resize((original_w, original_h), Image.NEAREST)
            pred_dir = os.path.join(test_save_path, "pred")
            os.makedirs(pred_dir, exist_ok=True)
            vis_resized.save(os.path.join(pred_dir, f"{case_name}_pred_unpad.png"))
        
        # 4) 保存去 padding 后的死亡概率分支图 (Death Probability Branch)
        if test_save_path and death_prob is not None:
            death_prob_chw = death_prob[0].cpu()  # [1, H, W]
            death_prob_unpad = unpad_tensor_chw(death_prob_chw, pad_info)  # [1, h', w']
            
            death_pm = death_prob_unpad[0].numpy()  # [h', w']
            
            # 映射回原始图像尺寸并保存
            death_pm_resized = Image.fromarray((death_pm * 255).astype(np.uint8)).resize(
                (original_w, original_h), Image.NEAREST
            )
            
            death_dir = os.path.join(test_save_path, "death_prob")
            os.makedirs(death_dir, exist_ok=True)
            death_pm_resized.save(os.path.join(death_dir, f"{case_name}_death_prob_unpad.png"))

        logging.info(f'Inference for case {case_name} completed (saved unpadded)')
    
    return "Inference Finished!"



# """
# z-score
# """
# def inference(args, model, test_save_path=None):
#     transform = ResizePadGenerator(output_size=(args.img_size, args.img_size))
#     db_test = PNGDataset(image_dir=args.volume_path, list_dir=args.list_dir, transform=transform)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

#     logging.info("{} test iterations".format(len(testloader)))
#     model.eval()

#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         image = sampled_batch["image"].cuda()             # [1, 1, H, W] (已 Z-score)
#         case_name = sampled_batch["case_name"][0]
#         pad_info = {k: v.item() for k, v in sampled_batch["pad_info"].items()} # DataLoader 出来的变 tensor 了，转回 int

#         with torch.no_grad():
#             # 这里的输出一定要和训练时一致 (model 返回两个值)
#             logits, _ = model(image)                     
#             probs = torch.softmax(logits, dim=1)         

#         # 保存输入图像 (反向映射回 0-255 用于查看)
#         inp_chw = image[0].cpu()                         
#         inp_unpad = unpad_tensor_chw(inp_chw, pad_info)   
#         # Z-score 后的图包含负数，这里转为 0-1 方便显示
#         display_img = (inp_unpad[0] - inp_unpad[0].min()) / (inp_unpad[0].max() - inp_unpad[0].min() + 1e-8)
#         input_img_np = (display_img.numpy() * 255).astype(np.uint8)
        
#         if test_save_path:
#             Image.fromarray(input_img_np).save(os.path.join(test_save_path, f"{case_name}_input_norm.png"))

#         # 2) 保存去 padding 后的每类概率图
#         probs_chw = probs[0].cpu()                       
#         probs_unpad = unpad_tensor_chw(probs_chw, pad_info)   

#         if test_save_path:
#             for cls_idx in range(probs_unpad.shape[0]):
#                 pm = probs_unpad[cls_idx].numpy()
#                 # 概率图已经是 0-1，直接存
#                 Image.fromarray((pm * 255).astype(np.uint8)).save(
#                     os.path.join(test_save_path, f"{case_name}_prob_class{cls_idx}.png")
#                 )

#         # 3) 保存去 padding 后的 argmax 预测图
#         pred_label = torch.argmax(probs_unpad, dim=0).cpu().numpy() # [h', w']
        
#         if test_save_path:
#             # 乘以 50 或 100 方便肉眼观察不同类别
#             vis = (pred_label * 100).astype(np.uint8)
#             Image.fromarray(vis).save(os.path.join(test_save_path, f"{case_name}_pred.png"))

#     return "Inference Finished!"






# setting the parameter
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928/20240926_a2780_post_drug__2024-09-26T22_59_12-Measurement/channel/more_bf/224_patches_all/brightfield', help='root dir for validation volume data')  # for acdc volume_path=root_dir
# parser.add_argument('--label_path', type=str, default='/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/datasets/Patches_datasets/labels', help='the root of label datasets')
parser.add_argument('--dataset', type=str,
                    default='/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/20250717/96_WM14_12Drugs/Patches/original', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
# parser.add_argument('--list_dir', type=str,
#                     default='./lists/list_one_drug_for_test/test.txt', help='list dir')
parser.add_argument('--list_dir', type=str,
                    default='/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928/20240926_a2780_post_drug__2024-09-26T22_59_12-Measurement/channel/more_bf/224_patches_all/brightfield/patches.txt', help='list dir')


parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

# parser.add_argument('--test_save_dir', type=str, default="",
#                     help='Path to save test results')




args = parser.parse_args()
config = get_config(args)

# #inference in test datasets
# def inference(args, model, test_save_path=None):
#     db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations per epoch".format(len(testloader)))
#     model.eval()
#     metric_list = 0.0
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         h, w = sampled_batch["image"].size()[1:]
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
#         metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
#                                       test_save_path=test_save_path, case=case_name)
#         metric_list += np.array(metric_i)
#         logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
#     metric_list = metric_list / len(db_test)
#     for i in range(1, args.num_classes):
#         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
#     performance = np.mean(metric_list, axis=0)[0]
#     mean_hd95 = np.mean(metric_list, axis=0)[1]
#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
#     return "Testing Finished!"




# main function
if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset_config = {
    #     'Synapse': {
    #         'Dataset': Synapse_dataset,
    #         'volume_path': args.volume_path,
    #         'list_dir': './lists/lists_Synapse',
    #         'num_classes': 9,
    #         'z_spacing': 1,
    #     },
    #     'Organoids': {
    #         'Dataset': Synapse_dataset,
    #         'volume_path': args.volume_path,
    #         'list_dir': 'args.list_dir',
    #         'num_classes': 3,
    #         'z_spacing': 1,
    #     },
    # }
    
    dataset_config = {
        'Organoids': {
            'Dataset': PNGDataset,
            'volume_path': args.volume_path,
            # 'label_path': args.label_path,
            'list_dir': args.list_dir,
            'num_classes': 3,
        },

    }
    

    
    
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    args.is_pretrain = True

    net = DoANet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, 'inference.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))

    # msg = net.load_state_dict(torch.load(snapshot))
    checkpoint = torch.load(snapshot, map_location="cpu")
    # state_dict = checkpoint
    # # 兼容不同保存格式
    # if isinstance(checkpoint, dict) and "model" in checkpoint:
    #     old_state_dict = checkpoint["model"]
    # elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    #     old_state_dict = checkpoint["state_dict"]
    # else:
    #     old_state_dict = checkpoint

    # new_state_dict = {}

    # for key, value in old_state_dict.items():
    #     new_key = key

    #     # 去掉 DataParallel 前缀
    #     if new_key.startswith("module."):
    #         new_key = new_key[len("module."):]

    #     # 旧 backbone encoder -> 新 encoder
    #     if new_key.startswith("swin_unet.patch_embed."):
    #         new_key = new_key.replace(
    #             "swin_unet.patch_embed.",
    #             "swin_unet.encoder.patch_embed.",
    #             1,
    #         )

    #     elif new_key == "swin_unet.absolute_pos_embed":
    #         new_key = "swin_unet.encoder.absolute_pos_embed"

    #     elif new_key.startswith("swin_unet.layers."):
    #         new_key = new_key.replace(
    #             "swin_unet.layers.",
    #             "swin_unet.encoder.layers.",
    #             1,
    #         )

    #     elif new_key.startswith("swin_unet.norm."):
    #         new_key = new_key.replace(
    #             "swin_unet.norm.",
    #             "swin_unet.encoder.norm.",
    #             1,
    #         )

    #     # 旧 decoder -> 新 decoder
    #     elif new_key.startswith("swin_unet.layers_up."):
    #         new_key = new_key.replace(
    #             "swin_unet.layers_up.",
    #             "swin_unet.decoder.layers_up.",
    #             1,
    #         )

    #     elif new_key.startswith("swin_unet.concat_back_dim."):
    #         new_key = new_key.replace(
    #             "swin_unet.concat_back_dim.",
    #             "swin_unet.decoder.concat_back_dim.",
    #             1,
    #         )

    #     elif new_key.startswith("swin_unet.norm_up."):
    #         new_key = new_key.replace(
    #             "swin_unet.norm_up.",
    #             "swin_unet.decoder.norm_up.",
    #             1,
    #         )

    #     elif new_key.startswith("swin_unet.up."):
    #         new_key = new_key.replace(
    #             "swin_unet.up.",
    #             "swin_unet.decoder.up.",
    #             1,
    #         )

    #     # seg_head 和 death_head 保持原名称
    #     new_state_dict[new_key] = value

    msg = net.load_state_dict(checkpoint, strict=False)

    print("Missing keys:")
    for key in msg.missing_keys:
        print("  ", key)

    print("Unexpected keys:")
    for key in msg.unexpected_keys:
        print("  ", key)
    # print("self trained swin unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        dataset_name = Path(args.volume_path).parents[3].name
        # print(f"dataname:{dataset_name}")
        args.test_save_dir = os.path.join(args.output_dir, "20260730",dataset_name)
        
        test_save_path = args.test_save_dir 
        # print(f"save_dir:{args.test_save_dir}")
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
        print(f"save_dir is none")
    inference(args, net, test_save_path)


