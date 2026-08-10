import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from networks.DoANet import DoANet
from config import get_config


class ResizePadGenerator:
    """Same preprocessing as the original inference code."""

    def __init__(self, output_size=(224, 224)):
        self.target_size = output_size[0]

    def __call__(self, image_np):
        image = Image.fromarray(image_np).convert("L")
        original_w, original_h = image.size

        scale = self.target_size / max(image.size)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        left = (self.target_size - new_w) // 2
        top = (self.target_size - new_h) // 2
        right = self.target_size - new_w - left
        bottom = self.target_size - new_h - top

        image = TF.pad(image, (left, top, right, bottom), fill=0)
        image = TF.to_tensor(image)

        return image, {
            "top": top,
            "left": left,
            "resized_h": new_h,
            "resized_w": new_w,
            "original_h": original_h,
            "original_w": original_w,
        }


def generate_patch_coords(image_shape, patch_size=224, stride=224):
    H, W = image_shape[:2]

    xs = list(range(0, max(W - patch_size + 1, 1), stride))
    ys = list(range(0, max(H - patch_size + 1, 1), stride))

    xs.append(max(W - patch_size, 0))
    ys.append(max(H - patch_size, 0))

    xs, ys = sorted(set(xs)), sorted(set(ys))

    return [
        (x, y, x + patch_size, y + patch_size)
        for y in ys
        for x in xs
    ]


def save_prob(prob, path):
    Image.fromarray(
        (np.clip(prob, 0, 1) * 255).astype(np.uint8)
    ).save(path)


def inference(
    image_path,
    model,
    save_dir,
    patch_size=224,
    stride=224,
    num_classes=3,
    death_logits=False,
):
    model.eval()

    pil_image = Image.open(image_path)
    raw = np.array(pil_image)

    print(
        f"Input: mode={pil_image.mode}, dtype={raw.dtype}, "
        f"min={raw.min()}, max={raw.max()}, mean={raw.mean():.2f}"
    )

    image = np.array(pil_image.convert("L"))
    H, W = image.shape
    name = Path(image_path).stem

    coords = generate_patch_coords(image.shape, patch_size, stride)
    transform = ResizePadGenerator((patch_size, patch_size))

    prob_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    death_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    
    # patch_dir = os.path.join(save_dir, "patches")
    # os.makedirs(patch_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (x1, y1, x2, y2) in enumerate(tqdm(coords)):
            patch = image[y1:y2, x1:x2]
            h, w = patch.shape

            # Image.fromarray(patch).save(
            #     os.path.join(
            #         patch_dir,
            #         f"{name}_patch_{i:02d}_x{x1}_y{y1}.png"
            #     )
            # )

            
            x, info = transform(patch)
            x = x.unsqueeze(0).cuda()

            logits, death = model(x)
            probs = torch.softmax(logits, dim=1)

            # patch-level debugging
            pred = probs[0].argmax(0)
            u, c = torch.unique(pred, return_counts=True)
            print(
                f"Patch {i:02d} {(x1, y1, x2, y2)} -> "
                f"{dict(zip(u.cpu().tolist(), c.cpu().tolist()))}"
            )

            top, left = info["top"], info["left"]
            rh, rw = info["resized_h"], info["resized_w"]

            probs = probs[0, :, top:top + rh, left:left + rw]

            if (rh, rw) != (h, w):
                probs = F.interpolate(
                    probs[None],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                )[0]

            prob_sum[:, y1:y1 + h, x1:x1 + w] += probs.cpu().numpy()

            if death is not None:
                if death_logits:
                    death = torch.sigmoid(death)

                death = death[0, 0] if death.ndim == 4 else death[0]
                death = death[top:top + rh, left:left + rw]

                if death.shape != (h, w):
                    death = F.interpolate(
                        death[None, None],
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False
                    )[0, 0]

                death_sum[y1:y1 + h, x1:x1 + w] += death.cpu().numpy()

            count[y1:y1 + h, x1:x1 + w] += 1

    if np.any(count == 0):
        raise RuntimeError("Some pixels are not covered.")

    prob = prob_sum / count[None]
    death_prob = death_sum / count
    pred = prob.argmax(0).astype(np.uint8)

    u, c = np.unique(pred, return_counts=True)
    print("Final labels:", dict(zip(u.tolist(), c.tolist())))

    for i in range(num_classes):
        p = prob[i]
        print(
            f"class {i}: "
            f"min={p.min():.3f}, max={p.max():.3f}, mean={p.mean():.3f}"
        )

    pred_dir = os.path.join(save_dir, "pred")
    prob_dir = os.path.join(save_dir, "prob")
    death_dir = os.path.join(save_dir, "death_prob")

    for folder in (pred_dir, prob_dir, death_dir):
        os.makedirs(folder, exist_ok=True)

    # raw class IDs: 0 / 1 / 2
    Image.fromarray(pred).save(
        os.path.join(pred_dir, f"{name}_pred_raw.png")
    )

    # visualization: 0 / 127 / 255
    vis = (pred / max(num_classes - 1, 1) * 255).astype(np.uint8)
    Image.fromarray(vis).save(
        os.path.join(pred_dir, f"{name}_pred.png")
    )

    for i in range(num_classes):
        save_prob(
            prob[i],
            os.path.join(prob_dir, f"{name}_prob_class{i}.png")
        )

    save_prob(
        death_prob,
        os.path.join(death_dir, f"{name}_death_prob.png")
    )


def load_model(args, config):
    model = DoANet(
        config,
        img_size=args.img_size,
        num_classes=args.num_classes
    ).cuda()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    msg = model.load_state_dict(checkpoint, strict=False)

    print(f"Missing keys: {len(msg.missing_keys)}")
    print(f"Unexpected keys: {len(msg.unexpected_keys)}")

    if msg.missing_keys:
        print("Missing:", msg.missing_keys)
    if msg.unexpected_keys:
        print("Unexpected:", msg.unexpected_keys)

    return model


parser = argparse.ArgumentParser()

parser.add_argument("--image_path", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--test_save_dir", default="./predictions")
parser.add_argument("--cfg", required=True)

parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--stride", type=int, default=224)
parser.add_argument("--num_classes", type=int, default=3)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--death_prob_is_logits", action="store_true")

# required by get_config()
parser.add_argument("--opts", nargs="+", default=None)
parser.add_argument("--zip", action="store_true")
parser.add_argument("--cache-mode", default="part")
parser.add_argument("--resume")
parser.add_argument("--accumulation-steps", type=int)
parser.add_argument("--use-checkpoint", action="store_true")
parser.add_argument("--amp-opt-level", default="O1")
parser.add_argument("--tag")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--throughput", action="store_true")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--max_epochs", type=int, default=150)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--dataset", default="Organoids")
parser.add_argument("--volume_path", default="")
parser.add_argument("--list_dir", default="")
parser.add_argument("--output_dir", default="")

args = parser.parse_args()
config = get_config(args)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = bool(args.deterministic)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = load_model(args, config)

    save_dir = os.path.join(
        args.test_save_dir,
        Path(args.image_path).stem
    )

    inference(
        image_path=args.image_path,
        model=model,
        save_dir=save_dir,
        patch_size=args.img_size,
        stride=args.stride,
        num_classes=args.num_classes,
        death_logits=args.death_prob_is_logits,
    )