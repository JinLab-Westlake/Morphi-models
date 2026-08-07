import argparse
import random
import sys
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from networks.DoANet import DoANet
from config import get_config
import numpy as np
from scipy.stats import pearsonr
from PIL import Image
import os
import logging
import torch
from tqdm import tqdm


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
                raise ValueError("transform must return (image, label, pad_info)")
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
    input: t [C,H,W] (Tensor)，output: to padding [C,h',w']。
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


def inference(args, model, test_save_path=None):
    transform = ResizePadGenerator(output_size=(args.img_size, args.img_size))

    # image_dir, label_dir, list_dir, transform
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

        # ========= Save the input image with the padding removed =========
        inp_chw = image[0]                            # [1, H, W]
        inp_unpad = unpad_tensor_chw(inp_chw, pad_info)   # [1, h', w']
        input_img_np = inp_unpad[0].cpu().numpy()

        # If the input is 0-1, map to 0-255.
        if input_img_np.max() <= 1.0:
            input_img_u8 = (input_img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            input_img_u8 = input_img_np.clip(0, 255).astype(np.uint8)

        if test_save_path:
            input_img_resized = Image.fromarray(input_img_u8).resize((original_w, original_h), Image.NEAREST)
            input_dir = os.path.join(test_save_path, "input")
            os.makedirs(input_dir, exist_ok=True)
            input_img_resized.save(os.path.join(input_dir, f"{case_name}_input_unpad.png"))

        # ========= Save the probability plot of each class after padding =========
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

        # ========= argmax pred =========
        pred_label = torch.argmax(probs[0], dim=0)   # [H, W], still padded
        pred_unpad = pred_label[
            pad_info["top"]: pred_label.shape[0] - pad_info["bottom"],
            pad_info["left"]: pred_label.shape[1] - pad_info["right"]
        ]                                            # [h', w']
        pred_np = pred_unpad.cpu().numpy().astype(np.uint8)

        # resize back to original size
        pred_resized = np.array(
            Image.fromarray(pred_np).resize((original_w, original_h), Image.NEAREST)
        ).astype(np.uint8)

        if test_save_path:
            vis = (pred_resized * (255 // max(1, probs.shape[1] - 1))).astype(np.uint8)
            pred_dir = os.path.join(test_save_path, "pred")
            os.makedirs(pred_dir, exist_ok=True)
            Image.fromarray(vis).save(os.path.join(pred_dir, f"{case_name}_pred_unpad.png"))

        # ========= Process the GT label and align to the original image size =========
        if isinstance(label, torch.Tensor):
            if label.ndim == 3:
                label_pad = label[0].cpu().numpy()
            else:
                label_pad = label.cpu().numpy()
        else:
            label_pad = np.array(label)

        # after padding
        gt_unpad = label_pad[
            pad_info["top"]: label_pad.shape[0] - pad_info["bottom"],
            pad_info["left"]: label_pad.shape[1] - pad_info["right"]
        ].astype(np.uint8)

        # resize
        gt_resized = np.array(
            Image.fromarray(gt_unpad).resize((original_w, original_h), Image.NEAREST)
        ).astype(np.uint8)

        # saving GT
        if test_save_path:
            gt_vis = (gt_resized * (255 // max(1, args.num_classes - 1))).astype(np.uint8)
            gt_dir = os.path.join(test_save_path, "gt")
            os.makedirs(gt_dir, exist_ok=True)
            Image.fromarray(gt_vis).save(os.path.join(gt_dir, f"{case_name}_gt_unpad.png"))

        # ========= Calculating metrics =========
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

    # ========= Summarize the full test set metrics =========
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

    net = DoANet(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

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