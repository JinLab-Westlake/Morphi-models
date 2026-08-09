import argparse
import logging
import os
import random
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.DoANet import DoANet
from config import get_config
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch


"""
without z-score
"""
class ResizePadGenerator:
    def __init__(self, output_size=(224, 224)):
        self.target_size = output_size[0]

    def __call__(self, image_np):
        # input image_np is numpy array (gray scale)
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
        # return (image Tensor, padding info)
        return image, pad_info


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
    input: t [C,H,W] (Tensor), output: after padding -> [C,h',w']。
    """
    top, bottom = pad_info["top"], pad_info["bottom"]
    left, right = pad_info["left"], pad_info["right"]
    _, H, W = t.shape
    h0, h1 = top, H - bottom
    w0, w1 = left, W - right
    return t[:, h0:h1, w0:w1]



def inference(args, model, test_save_path=None):
    transform = ResizePadGenerator(output_size=(args.img_size, args.img_size))
    db_test = PNGDataset(image_dir=args.volume_path, list_dir=args.list_dir, transform=transform)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image = sampled_batch["image"].cuda()             # [1, 1, H, W]
        case_name = sampled_batch["case_name"][0]
        case_name = os.path.splitext(os.path.basename(case_name))[0]
        # print(f"casename{case_name}")
        pad_info = sampled_batch["pad_info"][0] if isinstance(sampled_batch["pad_info"], list) else sampled_batch["pad_info"]
        
        # Assert pad_info is not None
        assert pad_info is not None, "pad_info is missing，please check if transform return (image, pad_info)"

        with torch.no_grad():
            # (death_prob_logits ot death_prob_map)
            logits, death_prob = model(image)  # [1, C, H, W] 和 [1, 1, H, W]
            probs = torch.softmax(logits, dim=1)


        # Save the input image with the padding removed
        inp_chw = image[0]  # [1, H, W]
        inp_unpad = unpad_tensor_chw(inp_chw, pad_info)  # [1, h', w']
        input_img_np = (inp_unpad[0].cpu().numpy() * 255).astype(np.uint8)
        
        if test_save_path:
            # Map back to original image dimensions via pad_info
            original_w, original_h = pad_info["original_w"], pad_info["original_h"]
            input_img_resized = Image.fromarray(input_img_np).resize((original_w, original_h), Image.NEAREST)
            input_dir = os.path.join(test_save_path, "input")
            os.makedirs(input_dir, exist_ok=True)
            input_img_resized.save(os.path.join(input_dir,f"{case_name}_input_unpad.png"))
            

        # Save the probability plot of each class after padding
        probs_chw = probs[0].cpu()  # [C, H, W]
        probs_unpad = unpad_tensor_chw(probs_chw, pad_info)  # [C, h', w']
        
        if test_save_path:
            C = probs_unpad.shape[0]
            for cls_idx in range(C):
                pm = probs_unpad[cls_idx].numpy()
                pm = pm / pm.max() if pm.max() > 0 else pm
                # Map back to original image dimensions and save
                pm_resized = Image.fromarray((pm * 255).astype(np.uint8)).resize((original_w, original_h), Image.NEAREST)
                prob_dir = os.path.join(test_save_path, "prob")
                os.makedirs(prob_dir, exist_ok=True)
                pm_resized.save(os.path.join(prob_dir, f"{case_name}_prob_class{cls_idx}_unpad.png"))

        # saving argmax pred
        pred_label = torch.argmax(probs[0], dim=0)  # [H, W]
        pred_unpad = pred_label[pad_info["top"]:pred_label.shape[0] - pad_info["bottom"],
                                 pad_info["left"]:pred_label.shape[1] - pad_info["right"]]  # [h', w']
        pred_np = pred_unpad.cpu().numpy()
        
        if test_save_path:
            # resize to original size
            vis = (pred_np * (255 // max(1, probs.shape[1] - 1))).astype(np.uint8)
            vis_resized = Image.fromarray(vis).resize((original_w, original_h), Image.NEAREST)
            pred_dir = os.path.join(test_save_path, "pred")
            os.makedirs(pred_dir, exist_ok=True)
            vis_resized.save(os.path.join(pred_dir, f"{case_name}_pred_unpad.png"))
        
        # saving to death prob map (Death Probability Branch)
        if test_save_path and death_prob is not None:
            death_prob_chw = death_prob[0].cpu()  # [1, H, W]
            death_prob_unpad = unpad_tensor_chw(death_prob_chw, pad_info)  # [1, h', w']
            
            death_pm = death_prob_unpad[0].numpy()  # [h', w']
            
            # Map back to original image dimensions and save
            death_pm_resized = Image.fromarray((death_pm * 255).astype(np.uint8)).resize(
                (original_w, original_h), Image.NEAREST
            )
            
            death_dir = os.path.join(test_save_path, "death_prob")
            os.makedirs(death_dir, exist_ok=True)
            death_pm_resized.save(os.path.join(death_dir, f"{case_name}_death_prob_unpad.png"))

        logging.info(f'Inference for case {case_name} completed (saved unpadded)')
    
    return "Inference Finished!"



# setting the parameter
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928/20240926_a2780_post_drug__2024-09-26T22_59_12-Measurement/channel/more_bf/224_patches_all/brightfield', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/20250717/96_WM14_12Drugs/Patches/original', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
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


args = parser.parse_args()
config = get_config(args)


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
    
    dataset_config = {
        'Organoids': {
            'Dataset': PNGDataset,
            'volume_path': args.volume_path,
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

    checkpoint = torch.load(snapshot, map_location="cpu")

    msg = net.load_state_dict(checkpoint, strict=False)

    print("Missing keys:")
    for key in msg.missing_keys:
        print("  ", key)

    print("Unexpected keys:")
    for key in msg.unexpected_keys:
        print("  ", key)
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


