# import argparse
# import logging
# import os
# import random
# import sys
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# # from datasets.dataset_synapse import Synapse_dataset
# from datasets.cell_datasets import Synapse_dataset
# from utils import test_single_volume
# from networks.vision_transformer import SwinUnet as ViT_seg
# from trainer import trainer_synapse
# from config import get_config
# from PIL import Image


# # define png datasets

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
    


# # val func

# import numpy as np
# from scipy.stats import pearsonr

# def safe_divide(a, b, eps=1e-8):
#     return a / (b + eps)

# def binary_metrics(pred_mask, gt_mask):
#     pred_mask = pred_mask.astype(np.uint8)
#     gt_mask = gt_mask.astype(np.uint8)

#     tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
#     fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
#     fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
#     tn = np.logical_and(pred_mask == 0, gt_mask == 0).sum()

#     dice = safe_divide(2 * tp, 2 * tp + fp + fn)
#     iou = safe_divide(tp, tp + fp + fn)
#     recall = safe_divide(tp, tp + fn)
#     precision = safe_divide(tp, tp + fp)
#     f1 = safe_divide(2 * precision * recall, precision + recall)

#     return {
#         "dice": dice,
#         "iou": iou,
#         "recall": recall,
#         "precision": precision,
#         "f1": f1
#     }


# def evaluate_single_case(pred, gt):
#     """
#     pred, gt: H x W, class map with
#     0=background, 1=live, 2=dead
#     """

#     # 1) foreground vs background
#     pred_fg = (pred > 0).astype(np.uint8)
#     gt_fg = (gt > 0).astype(np.uint8)
#     fg_metrics = binary_metrics(pred_fg, gt_fg)

#     # 2) live class
#     pred_live = (pred == 1).astype(np.uint8)
#     gt_live = (gt == 1).astype(np.uint8)
#     live_metrics = binary_metrics(pred_live, gt_live)

#     # 3) dead class
#     pred_dead = (pred == 2).astype(np.uint8)
#     gt_dead = (gt == 2).astype(np.uint8)
#     dead_metrics = binary_metrics(pred_dead, gt_dead)

#     # 4) ratio
#     pred_live_area = pred_live.sum()
#     pred_dead_area = pred_dead.sum()
#     gt_live_area = gt_live.sum()
#     gt_dead_area = gt_dead.sum()

#     pred_fg_area = pred_live_area + pred_dead_area
#     gt_fg_area = gt_live_area + gt_dead_area

#     pred_dead_ratio = np.nan if pred_fg_area == 0 else pred_dead_area / pred_fg_area
#     gt_dead_ratio = np.nan if gt_fg_area == 0 else gt_dead_area / gt_fg_area

#     return {
#         "dice_fg": fg_metrics["dice"],
#         "iou_fg": fg_metrics["iou"],
#         "recall_fg": fg_metrics["recall"],
#         "precision_fg": fg_metrics["precision"],
#         "f1_fg": fg_metrics["f1"],

#         "dice_live": live_metrics["dice"],
#         "iou_live": live_metrics["iou"],
#         "recall_live": live_metrics["recall"],
#         "precision_live": live_metrics["precision"],
#         "f1_live": live_metrics["f1"],

#         "dice_dead": dead_metrics["dice"],
#         "iou_dead": dead_metrics["iou"],
#         "recall_dead": dead_metrics["recall"],
#         "precision_dead": dead_metrics["precision"],
#         "f1_dead": dead_metrics["f1"],

#         "mdice_live_dead": (live_metrics["dice"] + dead_metrics["dice"]) / 2.0,
#         "miou_live_dead": (live_metrics["iou"] + dead_metrics["iou"]) / 2.0,

#         "gt_dead_ratio": gt_dead_ratio,
#         "pred_dead_ratio": pred_dead_ratio,
#     }    



# def inference(args,model,test_save_path = None):
#     # using the datasets self-defind
#     db_test = PNGDataset(image_dir = args.volume_path,label_dir = args.label_path, list_dir = args.list_dir)
#     testloader = DataLoader(db_test, batch_size =1, shuffle =False, num_workers =1)
#     logging.info("{} test iteration per epoch".format(len(testloader)))

#     model.eval()
#     metric_list = np.zeros((len(testloader), 6))  # 假设 `metric_i` 返回 6 个指标
#     # print(len(testloader))
    
#     # traverse all test samples
#     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
#         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch["case_name"]
#         h,w = image.size()[-2:]
        
#         #using inference function to calculate the scale
#         metric_i = test_single_volume(image,label,model,classes = args.num_classes,
#                                       patch_size = [args.img_size,args.img_size],test_save_path=test_save_path,case =case_name)
#         # metric_list += np.array(metric_i)
        
#         metric_i = np.array(metric_i)
#         if metric_i.size != metric_list.shape[1]:  # 如果 `metric_i` 和 `metric_list` 形状不匹配
#             metric_i = np.resize(metric_i, (1, 6))  # 调整为 (1, 6)
#         metric_list[i_batch] = metric_i  # 将当前批次的结果累加到 `metric_list` 中

#         logging.info('idx %d case %s mean_dice %f mean_hd95 %f mIOU %f recall %f precision %f f1 %f ' %
#                      (i_batch, case_name, np.mean(metric_i,axis=0)[0],np.mean(metric_i,axis=0)[1],np.mean(metric_i,axis=0)[2],np.mean(metric_i,axis=0)[3],np.mean(metric_i,axis=0)[4],np.mean(metric_i,axis=0)[5]))
    
#     metric_list = metric_list/len(db_test)
#     for i in range(1, args.num_classes):
#         logging.info('mean class %d mean_dice  %f mean_hd95 %f mIOU %f recall %f precision %f f1 %f' %(i, metric_list[i-1][0],metric_list[i-1][1],metric_list[i-1][2],metric_list[i-1][3],metric_list[i-1][4],metric_list[i-1][5]))
    
#     performance = np.mean(metric_list, axis = 0)[0]
#     mean_hd95 = np.mean(metric_list, axis =0) [1]
#     mIOU = np.mean(metric_list, axis =0) [2]
#     recall = np.mean(metric_list, axis =0) [3]
#     precision = np.mean(metric_list, axis =0) [4]
#     f1 = np.mean(metric_list, axis =0) [5]

#     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mIOU %f recall %f precision %f f1 %f' % (performance, mean_hd95,mIOU,recall,precision,f1))
    
#     return "Testing Finished!"








# # setting the parameter
# parser = argparse.ArgumentParser()
# parser.add_argument('--volume_path', type=str,
#                     default='../data/Patches_datasets/test/images', help='root dir for validation volume data')  # for acdc volume_path=root_dir
# parser.add_argument('--label_path', type=str, default='/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/datasets/Patches_datasets/labels', help='the root of label datasets')
# parser.add_argument('--dataset', type=str,
#                     default='Patches_datasets/test', help='experiment_name')
# parser.add_argument('--num_classes', type=int,
#                     default=3, help='output channel of network')
# parser.add_argument('--list_dir', type=str,
#                     default='./lists/list_patches/test.txt', help='list dir')
# parser.add_argument('--output_dir', type=str, help='output dir')   
# parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
# parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
# parser.add_argument('--batch_size', type=int, default=24,
#                     help='batch_size per gpu')
# parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
# parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
# parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
# parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
# parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
# parser.add_argument('--seed', type=int, default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
# parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+',
#     )
# parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
# parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
#                     help='no: no cache, '
#                             'full: cache all data, '
#                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
# parser.add_argument('--resume', help='resume from checkpoint')
# parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
# parser.add_argument('--use-checkpoint', action='store_true',
#                     help="whether to use gradient checkpointing to save memory")
# parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
#                     help='mixed precision opt level, if O0, no amp is used')
# parser.add_argument('--tag', help='tag of experiment')
# parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
# parser.add_argument('--throughput', action='store_true', help='Test throughput only')

# args = parser.parse_args()
# config = get_config(args)

# # #inference in test datasets
# # def inference(args, model, test_save_path=None):
# #     db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)
# #     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
# #     logging.info("{} test iterations per epoch".format(len(testloader)))
# #     model.eval()
# #     metric_list = 0.0
# #     for i_batch, sampled_batch in tqdm(enumerate(testloader)):
# #         h, w = sampled_batch["image"].size()[1:]
# #         image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
# #         metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
# #                                       test_save_path=test_save_path, case=case_name)
# #         metric_list += np.array(metric_i)
# #         logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
# #     metric_list = metric_list / len(db_test)
# #     for i in range(1, args.num_classes):
# #         logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
# #     performance = np.mean(metric_list, axis=0)[0]
# #     mean_hd95 = np.mean(metric_list, axis=0)[1]
# #     logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
# #     return "Testing Finished!"




# # main function
# if __name__ == "__main__":

#     if not args.deterministic:
#         cudnn.benchmark = True
#         cudnn.deterministic = False
#     else:
#         cudnn.benchmark = False
#         cudnn.deterministic = True
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)

#     # dataset_config = {
#     #     'Synapse': {
#     #         'Dataset': Synapse_dataset,
#     #         'volume_path': args.volume_path,
#     #         'list_dir': './lists/lists_Synapse',
#     #         'num_classes': 9,
#     #         'z_spacing': 1,
#     #     },
#     #     'Organoids': {
#     #         'Dataset': Synapse_dataset,
#     #         'volume_path': args.volume_path,
#     #         'list_dir': './lists/lists_organoids',
#     #         'num_classes': 2,
#     #         'z_spacing': 1,
#     #     },
#     # }
    
#     dataset_config = {
#         'Organoids': {
#             'Dataset': PNGDataset,
#             'volume_path': args.volume_path,
#             'label_path': args.label_path,
#             'list_dir': args.list_dir,
#             'num_classes': 3,
#         },
#     }
    
#     dataset_name = args.dataset
#     args.num_classes = dataset_config[dataset_name]['num_classes']
#     args.volume_path = dataset_config[dataset_name]['volume_path']
#     args.Dataset = dataset_config[dataset_name]['Dataset']
#     args.list_dir = dataset_config[dataset_name]['list_dir']
#     args.label_path = dataset_config[dataset_name]['label_path']
#     # args.z_spacing = dataset_config[dataset_name]['z_spacing']
#     args.is_pretrain = True

#     net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

#     snapshot = os.path.join(args.output_dir, 'best_model.pth')
#     if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
#     # if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(799)
    
#     # if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
#     msg = net.load_state_dict(torch.load(snapshot))
#     print("self trained swin unet",msg)
#     snapshot_name = snapshot.split('/')[-1]

#     log_folder = './test_log/test_log_'
#     os.makedirs(log_folder, exist_ok=True)
#     logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#     logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#     logging.info(str(args))
#     logging.info(snapshot_name)

#     if args.is_savenii:
#         args.test_save_dir = os.path.join(args.output_dir, "predictions/20241021")
#         test_save_path = args.test_save_dir 
#         os.makedirs(test_save_path, exist_ok=True)
#     else:
#         test_save_path = None
#     inference(args, net, test_save_path)





import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from PIL import Image

from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config


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

#         image = np.array(Image.open(image_path).convert('L'))

#         pad_info = None
#         if self.transform:
#             out = self.transform(image)
#             if isinstance(out, tuple):
#                 image, pad_info = out
#             else:
#                 image = out

#         return {"image": image, "case_name": case_name, "pad_info": pad_info}

class PNGDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, list_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        with open(list_dir, 'r') as f:
            self.image_list = [line.strip() for line in f]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        case_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, case_name)
        label_path = os.path.join(self.label_dir, case_name)

        image = np.array(Image.open(image_path).convert('L'))
        label = np.array(Image.open(label_path).convert('L'))

        pad_info = None
        if self.transform:
            out = self.transform(image, label)
            if isinstance(out, tuple) and len(out) == 3:
                image, label, pad_info = out
            else:
                raise ValueError("transform 必须返回 (image, label, pad_info)")
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
            label = torch.from_numpy(label).long()
            h, w = image.shape[-2:]
            pad_info = {
                "top": 0,
                "bottom": 0,
                "left": 0,
                "right": 0,
                "original_h": h,
                "original_w": w,
            }

        return {
            "image": image,
            "label": label,
            "case_name": case_name,
            "pad_info": pad_info
        }



"""
without z-score
"""
class ResizePadGenerator:
    def __init__(self, output_size):
        self.output_size = output_size  # (H, W)

    def __call__(self, image, label):
        original_h, original_w = image.shape[:2]
        out_h, out_w = self.output_size

        scale = min(out_w / original_w, out_h / original_h)
        new_w = int(round(original_w * scale))
        new_h = int(round(original_h * scale))

        image_resized = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
        label_resized = np.array(Image.fromarray(label).resize((new_w, new_h), Image.NEAREST))

        pad_h = out_h - new_h
        pad_w = out_w - new_w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        image_padded = np.pad(image_resized, ((top, bottom), (left, right)), mode='constant')
        label_padded = np.pad(label_resized, ((top, bottom), (left, right)), mode='constant')

        image_tensor = torch.from_numpy(image_padded).unsqueeze(0).float() / 255.0
        label_tensor = torch.from_numpy(label_padded).long()

        pad_info = {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "original_h": original_h,
            "original_w": original_w,
        }

        return image_tensor, label_tensor, pad_info
    
    
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




def safe_mean(metric_list, idx):
    if len(metric_list) == 0:
        return np.nan
    arr = np.array(metric_list, dtype=np.float64)
    return np.nanmean(arr[:, idx])


import numpy as np
from scipy.stats import pearsonr
from PIL import Image
import os
import logging
import torch
from tqdm import tqdm


def safe_divide(a, b, eps=1e-8):
    return a / (b + eps)


def binary_metrics(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)

    tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
    fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()

    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    recall = safe_divide(tp, tp + fn)
    precision = safe_divide(tp, tp + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "dice": dice,
        "iou": iou,
        "recall": recall,
        "precision": precision,
        "f1": f1
    }


def evaluate_single_case(pred, gt):
    """
    pred, gt: H x W
    class map:
        0 = background
        1 = live
        2 = dead
    """

    pred_fg = (pred > 0).astype(np.uint8)
    gt_fg = (gt > 0).astype(np.uint8)
    fg_metrics = binary_metrics(pred_fg, gt_fg)

    pred_live = (pred == 1).astype(np.uint8)
    gt_live = (gt == 1).astype(np.uint8)
    live_metrics = binary_metrics(pred_live, gt_live)

    pred_dead = (pred == 2).astype(np.uint8)
    gt_dead = (gt == 2).astype(np.uint8)
    dead_metrics = binary_metrics(pred_dead, gt_dead)

    pred_live_area = pred_live.sum()
    pred_dead_area = pred_dead.sum()
    gt_live_area = gt_live.sum()
    gt_dead_area = gt_dead.sum()

    pred_fg_area = pred_live_area + pred_dead_area
    gt_fg_area = gt_live_area + gt_dead_area

    pred_dead_ratio = np.nan if pred_fg_area == 0 else pred_dead_area / pred_fg_area
    gt_dead_ratio = np.nan if gt_fg_area == 0 else gt_dead_area / gt_fg_area

    return {
        "dice_fg": fg_metrics["dice"],
        "iou_fg": fg_metrics["iou"],
        "recall_fg": fg_metrics["recall"],
        "precision_fg": fg_metrics["precision"],
        "f1_fg": fg_metrics["f1"],

        "dice_live": live_metrics["dice"],
        "iou_live": live_metrics["iou"],
        "recall_live": live_metrics["recall"],
        "precision_live": live_metrics["precision"],
        "f1_live": live_metrics["f1"],

        "dice_dead": dead_metrics["dice"],
        "iou_dead": dead_metrics["iou"],
        "recall_dead": dead_metrics["recall"],
        "precision_dead": dead_metrics["precision"],
        "f1_dead": dead_metrics["f1"],

        "mdice_live_dead": (live_metrics["dice"] + dead_metrics["dice"]) / 2.0,
        "miou_live_dead": (live_metrics["iou"] + dead_metrics["iou"]) / 2.0,

        "gt_dead_ratio": gt_dead_ratio,
        "pred_dead_ratio": pred_dead_ratio,
    }


# def inference(args, model, test_save_path=None):
#     transform = ResizePadGenerator(output_size=(args.img_size, args.img_size))
#     db_test = PNGDataset(
#         image_dir=args.volume_path,
#         label_dir=args.label_path,
#         list_dir=args.list_dir,
#         transform=transform
#     )
#     testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
#     logging.info("{} test iterations".format(len(testloader)))

#     model.eval()

#     fg_metric_list = []
#     live_metric_list = []
#     dead_metric_list = []

#     ratio_true_list = []
#     ratio_pred_list = []

#     for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
#         image = sampled_batch["image"]
#         label = sampled_batch["label"]
#         case_name = sampled_batch["case_name"][0]

#         result = test_single_volume(
#             image,
#             label,
#             model,
#             classes=args.num_classes,
#             patch_size=[args.img_size, args.img_size],
#             test_save_path=test_save_path,
#             case=case_name
#         )

#         fg_metrics = result["fg_metrics"]
#         class_metrics = result["class_metrics"]

#         live_metrics = class_metrics.get(1, [np.nan] * 6)
#         dead_metrics = class_metrics.get(2, [np.nan] * 6)

#         fg_metric_list.append(fg_metrics)
#         live_metric_list.append(live_metrics)
#         dead_metric_list.append(dead_metrics)

#         gt_ratio = result["gt_dead_ratio"]
#         pred_ratio = result["pred_dead_ratio"]

#         if not np.isnan(gt_ratio) and not np.isnan(pred_ratio):
#             ratio_true_list.append(gt_ratio)
#             ratio_pred_list.append(pred_ratio)

#         logging.info(
#             "idx %d case %s | "
#             "FG Dice %.4f HD95 %.4f IoU %.4f Recall %.4f Precision %.4f F1 %.4f | "
#             "Live Dice %.4f | Dead Dice %.4f | "
#             "GT dead ratio %.4f | Pred dead ratio %.4f"
#             % (
#                 i_batch,
#                 case_name,
#                 fg_metrics[0], fg_metrics[1], fg_metrics[2], fg_metrics[3], fg_metrics[4], fg_metrics[5],
#                 live_metrics[0],
#                 dead_metrics[0],
#                 gt_ratio if not np.isnan(gt_ratio) else -1,
#                 pred_ratio if not np.isnan(pred_ratio) else -1
#             )
#         )

#     fg_metric_list = np.array(fg_metric_list, dtype=np.float64)
#     live_metric_list = np.array(live_metric_list, dtype=np.float64)
#     dead_metric_list = np.array(dead_metric_list, dtype=np.float64)

#     mean_fg = np.nanmean(fg_metric_list, axis=0) if len(fg_metric_list) > 0 else [np.nan] * 6
#     mean_live = np.nanmean(live_metric_list, axis=0) if len(live_metric_list) > 0 else [np.nan] * 6
#     mean_dead = np.nanmean(dead_metric_list, axis=0) if len(dead_metric_list) > 0 else [np.nan] * 6

#     mean_ld_dice = np.nanmean([mean_live[0], mean_dead[0]])
#     mean_ld_iou = np.nanmean([mean_live[2], mean_dead[2]])
#     mean_ld_recall = np.nanmean([mean_live[3], mean_dead[3]])
#     mean_ld_precision = np.nanmean([mean_live[4], mean_dead[4]])
#     mean_ld_f1 = np.nanmean([mean_live[5], mean_dead[5]])

#     if len(ratio_true_list) > 1:
#         ratio_true = np.array(ratio_true_list, dtype=np.float64)
#         ratio_pred = np.array(ratio_pred_list, dtype=np.float64)

#         mae_ratio = np.mean(np.abs(ratio_pred - ratio_true))
#         rmse_ratio = np.sqrt(np.mean((ratio_pred - ratio_true) ** 2))
#         pearson_r, pearson_p = pearsonr(ratio_true, ratio_pred)
#     else:
#         mae_ratio = np.nan
#         rmse_ratio = np.nan
#         pearson_r = np.nan
#         pearson_p = np.nan

#     logging.info("========== Final Results ==========")

#     logging.info(
#         "Foreground: Dice %.4f | HD95 %.4f | IoU %.4f | Recall %.4f | Precision %.4f | F1 %.4f"
#         % (mean_fg[0], mean_fg[1], mean_fg[2], mean_fg[3], mean_fg[4], mean_fg[5])
#     )

#     logging.info(
#         "Live:       Dice %.4f | HD95 %.4f | IoU %.4f | Recall %.4f | Precision %.4f | F1 %.4f"
#         % (mean_live[0], mean_live[1], mean_live[2], mean_live[3], mean_live[4], mean_live[5])
#     )

#     logging.info(
#         "Dead:       Dice %.4f | HD95 %.4f | IoU %.4f | Recall %.4f | Precision %.4f | F1 %.4f"
#         % (mean_dead[0], mean_dead[1], mean_dead[2], mean_dead[3], mean_dead[4], mean_dead[5])
#     )

#     logging.info(
#         "Live/Dead Mean: Dice %.4f | IoU %.4f | Recall %.4f | Precision %.4f | F1 %.4f"
#         % (mean_ld_dice, mean_ld_iou, mean_ld_recall, mean_ld_precision, mean_ld_f1)
#     )

#     logging.info(
#         "Dead Ratio: MAE %.4f | RMSE %.4f | Pearson r %.4f | p %.4e"
#         % (mae_ratio, rmse_ratio, pearson_r, pearson_p)
#     )

#     return {
#         "dice_fg": mean_fg[0],
#         "iou_fg": mean_fg[2],
#         "dice_live": mean_live[0],
#         "dice_dead": mean_dead[0],
#         "mdice_live_dead": mean_ld_dice,
#         "mae_ratio": mae_ratio,
#         "rmse_ratio": rmse_ratio,
#         "pearson_r": pearson_r,
#         "pearson_p": pearson_p,
#     }


def inference(args, model, test_save_path=None):
    transform = ResizePadGenerator(output_size=(args.img_size, args.img_size))

    # 这里假设你的 PNGDataset 现在支持 image_dir, label_dir, list_dir, transform
    db_test = PNGDataset(
        image_dir=args.volume_path,
        label_dir=args.label_path,
        list_dir=args.list_dir,
        transform=transform
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info("{} test iterations".format(len(testloader)))
    model.eval()

    all_metrics = []
    ratio_true_list = []
    ratio_pred_list = []

    for i_batch, sampled_batch in tqdm(enumerate(testloader), total=len(testloader)):
        image = sampled_batch["image"].cuda()         # [1, 1, H, W], padded
        label = sampled_batch["label"]                # [1, H, W], padded or aligned with transform
        case_name = sampled_batch["case_name"][0]
        case_name = os.path.splitext(os.path.basename(case_name))[0]

        pad_info = sampled_batch["pad_info"][0] if isinstance(sampled_batch["pad_info"], list) else sampled_batch["pad_info"]
        assert pad_info is not None, "pad_info 缺失，请确认 transform 返回 (image, label, pad_info) 或 (image, pad_info)"

        original_w = int(pad_info["original_w"])
        original_h = int(pad_info["original_h"])

        with torch.no_grad():
            outputs = model(image)
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)      # [1, C, H, W]

        # ========= 1) 保存去padding后的输入图像 =========
        inp_chw = image[0]                            # [1, H, W]
        inp_unpad = unpad_tensor_chw(inp_chw, pad_info)   # [1, h', w']
        input_img_np = inp_unpad[0].cpu().numpy()

        # 如果输入是 0~1，映射到 0~255；否则做稳妥截断
        if input_img_np.max() <= 1.0:
            input_img_u8 = (input_img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            input_img_u8 = input_img_np.clip(0, 255).astype(np.uint8)

        if test_save_path:
            input_img_resized = Image.fromarray(input_img_u8).resize((original_w, original_h), Image.NEAREST)
            input_dir = os.path.join(test_save_path, "input")
            os.makedirs(input_dir, exist_ok=True)
            input_img_resized.save(os.path.join(input_dir, f"{case_name}_input_unpad.png"))

        # ========= 2) 保存去padding后的每类概率图 =========
        probs_chw = probs[0].cpu()                    # [C, H, W]
        probs_unpad = unpad_tensor_chw(probs_chw, pad_info)   # [C, h', w']

        if test_save_path:
            prob_dir = os.path.join(test_save_path, "prob")
            os.makedirs(prob_dir, exist_ok=True)

            C = probs_unpad.shape[0]
            for cls_idx in range(C):
                pm = probs_unpad[cls_idx].numpy()
                pm = pm / pm.max() if pm.max() > 0 else pm
                pm_u8 = (pm * 255).astype(np.uint8)
                pm_resized = Image.fromarray(pm_u8).resize((original_w, original_h), Image.NEAREST)
                pm_resized.save(os.path.join(prob_dir, f"{case_name}_prob_class{cls_idx}_unpad.png"))

        # ========= 3) 保存去padding后的 argmax 预测图 =========
        pred_label = torch.argmax(probs[0], dim=0)   # [H, W], still padded
        pred_unpad = pred_label[
            pad_info["top"]: pred_label.shape[0] - pad_info["bottom"],
            pad_info["left"]: pred_label.shape[1] - pad_info["right"]
        ]                                            # [h', w']
        pred_np = pred_unpad.cpu().numpy().astype(np.uint8)

        # resize 回原图大小，作为最终评估和保存使用
        pred_resized = np.array(
            Image.fromarray(pred_np).resize((original_w, original_h), Image.NEAREST)
        ).astype(np.uint8)

        if test_save_path:
            vis = (pred_resized * (255 // max(1, probs.shape[1] - 1))).astype(np.uint8)
            pred_dir = os.path.join(test_save_path, "pred")
            os.makedirs(pred_dir, exist_ok=True)
            Image.fromarray(vis).save(os.path.join(pred_dir, f"{case_name}_pred_unpad.png"))

        # ========= 4) 处理 GT label，对齐到原图大小 =========
        # 这里假设 dataset 返回的 label 和 image 一起经过了同样的 ResizePadGenerator
        if isinstance(label, torch.Tensor):
            if label.ndim == 3:
                label_pad = label[0].cpu().numpy()
            else:
                label_pad = label.cpu().numpy()
        else:
            label_pad = np.array(label)

        # 去 padding
        gt_unpad = label_pad[
            pad_info["top"]: label_pad.shape[0] - pad_info["bottom"],
            pad_info["left"]: label_pad.shape[1] - pad_info["right"]
        ].astype(np.uint8)

        # resize 回原图大小
        gt_resized = np.array(
            Image.fromarray(gt_unpad).resize((original_w, original_h), Image.NEAREST)
        ).astype(np.uint8)

        # 可选保存 GT
        if test_save_path:
            gt_vis = (gt_resized * (255 // max(1, args.num_classes - 1))).astype(np.uint8)
            gt_dir = os.path.join(test_save_path, "gt")
            os.makedirs(gt_dir, exist_ok=True)
            Image.fromarray(gt_vis).save(os.path.join(gt_dir, f"{case_name}_gt_unpad.png"))

        # ========= 5) 计算指标 =========
        case_metrics = evaluate_single_case(pred_resized, gt_resized)
        all_metrics.append(case_metrics)

        gt_ratio = case_metrics["gt_dead_ratio"]
        pred_ratio = case_metrics["pred_dead_ratio"]
        if not np.isnan(gt_ratio) and not np.isnan(pred_ratio):
            ratio_true_list.append(gt_ratio)
            ratio_pred_list.append(pred_ratio)

        logging.info(
            "idx %d case %s | "
            "FG Dice %.4f IoU %.4f Recall %.4f Precision %.4f F1 %.4f | "
            "Live Dice %.4f IoU %.4f | "
            "Dead Dice %.4f IoU %.4f | "
            "GT dead ratio %.4f | Pred dead ratio %.4f"
            % (
                i_batch,
                case_name,
                case_metrics["dice_fg"],
                case_metrics["iou_fg"],
                case_metrics["recall_fg"],
                case_metrics["precision_fg"],
                case_metrics["f1_fg"],
                case_metrics["dice_live"],
                case_metrics["iou_live"],
                case_metrics["dice_dead"],
                case_metrics["iou_dead"],
                gt_ratio if not np.isnan(gt_ratio) else -1,
                pred_ratio if not np.isnan(pred_ratio) else -1
            )
        )

        logging.info(f'Inference for case {case_name} completed (saved unpadded + evaluated)')

    # ========= 6) 汇总全测试集指标 =========
    mean_dice_fg = np.mean([m["dice_fg"] for m in all_metrics]) if all_metrics else np.nan
    mean_iou_fg = np.mean([m["iou_fg"] for m in all_metrics]) if all_metrics else np.nan
    mean_recall_fg = np.mean([m["recall_fg"] for m in all_metrics]) if all_metrics else np.nan
    mean_precision_fg = np.mean([m["precision_fg"] for m in all_metrics]) if all_metrics else np.nan
    mean_f1_fg = np.mean([m["f1_fg"] for m in all_metrics]) if all_metrics else np.nan

    mean_dice_live = np.mean([m["dice_live"] for m in all_metrics]) if all_metrics else np.nan
    mean_iou_live = np.mean([m["iou_live"] for m in all_metrics]) if all_metrics else np.nan
    mean_recall_live = np.mean([m["recall_live"] for m in all_metrics]) if all_metrics else np.nan
    mean_precision_live = np.mean([m["precision_live"] for m in all_metrics]) if all_metrics else np.nan
    mean_f1_live = np.mean([m["f1_live"] for m in all_metrics]) if all_metrics else np.nan

    mean_dice_dead = np.mean([m["dice_dead"] for m in all_metrics]) if all_metrics else np.nan
    mean_iou_dead = np.mean([m["iou_dead"] for m in all_metrics]) if all_metrics else np.nan
    mean_recall_dead = np.mean([m["recall_dead"] for m in all_metrics]) if all_metrics else np.nan
    mean_precision_dead = np.mean([m["precision_dead"] for m in all_metrics]) if all_metrics else np.nan
    mean_f1_dead = np.mean([m["f1_dead"] for m in all_metrics]) if all_metrics else np.nan

    mean_mdice_live_dead = np.mean([m["mdice_live_dead"] for m in all_metrics]) if all_metrics else np.nan
    mean_miou_live_dead = np.mean([m["miou_live_dead"] for m in all_metrics]) if all_metrics else np.nan

    if len(ratio_true_list) > 1:
        ratio_true = np.array(ratio_true_list, dtype=np.float64)
        ratio_pred = np.array(ratio_pred_list, dtype=np.float64)
        mae_ratio = np.mean(np.abs(ratio_pred - ratio_true))
        rmse_ratio = np.sqrt(np.mean((ratio_pred - ratio_true) ** 2))
        pearson_r, pearson_p = pearsonr(ratio_true, ratio_pred)
    else:
        mae_ratio = np.nan
        rmse_ratio = np.nan
        pearson_r = np.nan
        pearson_p = np.nan

    logging.info("========== Final Results ==========")
    logging.info(
        "Foreground: Dice %.4f | IoU %.4f | Recall %.4f | Precision %.4f | F1 %.4f"
        % (mean_dice_fg, mean_iou_fg, mean_recall_fg, mean_precision_fg, mean_f1_fg)
    )
    logging.info(
        "Live:       Dice %.4f | IoU %.4f | Recall %.4f | Precision %.4f | F1 %.4f"
        % (mean_dice_live, mean_iou_live, mean_recall_live, mean_precision_live, mean_f1_live)
    )
    logging.info(
        "Dead:       Dice %.4f | IoU %.4f | Recall %.4f | Precision %.4f | F1 %.4f"
        % (mean_dice_dead, mean_iou_dead, mean_recall_dead, mean_precision_dead, mean_f1_dead)
    )
    logging.info(
        "Live/Dead Mean: Dice %.4f | IoU %.4f"
        % (mean_mdice_live_dead, mean_miou_live_dead)
    )
    logging.info(
        "Dead Ratio: MAE %.4f | RMSE %.4f | Pearson r %.4f | p %.4e"
        % (mae_ratio, rmse_ratio, pearson_r, pearson_p)
    )

    return {
        "dice_fg": mean_dice_fg,
        "iou_fg": mean_iou_fg,
        "dice_live": mean_dice_live,
        "iou_live": mean_iou_live,
        "dice_dead": mean_dice_dead,
        "iou_dead": mean_iou_dead,
        "mdice_live_dead": mean_mdice_live_dead,
        "miou_live_dead": mean_miou_live_dead,
        "mae_ratio": mae_ratio,
        "rmse_ratio": rmse_ratio,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    '--volume_path',
    type=str,
    default='../data/Patches_datasets/test/images',
    help='root dir for test image data'
)
parser.add_argument(
    '--label_path',
    type=str,
    default='/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/datasets/Patches_datasets/train/labels',
    help='root dir for test label data'
)
parser.add_argument(
    '--dataset',
    type=str,
    default='Organoids',
    help='dataset name'
)
parser.add_argument(
    '--num_classes',
    type=int,
    default=3,
    help='output channel of network'
)
parser.add_argument(
    '--list_dir',
    type=str,
    default='/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/datasets/Patches_datasets/train/train.txt',
    help='test list txt'
)
parser.add_argument('--output_dir', type=str, required=True, help='model output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size')
parser.add_argument('--is_savenii', action="store_true", help='whether to save prediction results')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='prediction save dir')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs.",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument(
    '--cache-mode',
    type=str,
    default='part',
    choices=['no', 'full', 'part'],
    help='no: no cache, full: cache all data, part: cache one shard'
)
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing")
parser.add_argument(
    '--amp-opt-level',
    type=str,
    default='O1',
    choices=['O0', 'O1', 'O2'],
    help='mixed precision opt level'
)
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


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

    dataset_config = {
        'Organoids': {
            'Dataset': PNGDataset,
            'volume_path': args.volume_path,
            'label_path': args.label_path,
            'list_dir': args.list_dir,
            'num_classes': 3,
        },
    }

    dataset_name = args.dataset
    if dataset_name not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(dataset_config.keys())}")

    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.label_path = dataset_config[dataset_name]['label_path']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot):
        snapshot = snapshot.replace('best_model.pth', 'epoch_' + str(args.max_epochs - 1) + '.pth')

    # msg = net.load_state_dict(torch.load(snapshot))
    msg = net.load_state_dict(torch.load(snapshot, map_location='cuda'))
    print("self trained swin unet", msg)

    snapshot_name = os.path.basename(snapshot)

    log_folder = './test_log'
    os.makedirs(log_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_folder, snapshot_name + ".txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions_train")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    results = inference(args, net, test_save_path)
    logging.info("Returned summary: %s", str(results))