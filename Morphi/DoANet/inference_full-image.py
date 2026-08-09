import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from networks.DoANet import DoANet
from config import get_config


# ============================================================
# Patch coordinate generation
# ============================================================

def generate_patch_coords(image_shape, patch_size=224, stride=224):
    """
    Generate patch coordinates covering the whole image.

    Parameters
    ----------
    image_shape : tuple
        Image shape, e.g. (H, W) or (H, W, C).

    patch_size : int
        Patch size.

    stride : int
        Sliding stride.

    Returns
    -------
    coords : list
        [(x1, y1, x2, y2), ...]

    Example
    -------
    For a 1080x1080 image, patch_size=224, stride=224:

    xs = [0, 224, 448, 672, 856]
    ys = [0, 224, 448, 672, 856]

    Total patches = 25.

    The last patch overlaps with the previous patch by 40 px.
    """

    H, W = image_shape[:2]

    coords = []

    xs = list(
        range(
            0,
            max(W - patch_size + 1, 1),
            stride
        )
    )

    ys = list(
        range(
            0,
            max(H - patch_size + 1, 1),
            stride
        )
    )

    # --------------------------------------------------------
    # Make sure right boundary is covered
    # --------------------------------------------------------
    if len(xs) == 0 or xs[-1] != W - patch_size:
        xs.append(
            max(W - patch_size, 0)
        )

    # --------------------------------------------------------
    # Make sure bottom boundary is covered
    # --------------------------------------------------------
    if len(ys) == 0 or ys[-1] != H - patch_size:
        ys.append(
            max(H - patch_size, 0)
        )

    # remove duplicated coordinates
    xs = sorted(set(xs))
    ys = sorted(set(ys))

    for y1 in ys:
        for x1 in xs:

            x2 = x1 + patch_size
            y2 = y1 + patch_size

            coords.append(
                (x1, y1, x2, y2)
            )

    return coords


# ============================================================
# Convert image patch to model tensor
# ============================================================

def patch_to_tensor(
    patch_np,
    patch_size=224,
    device="cuda"
):
    """
    Convert grayscale numpy patch to model input tensor.

    Input
    -----
    patch_np:
        [H, W], uint8

    Output
    ------
    tensor:
        [1, 1, patch_size, patch_size]

    valid_h, valid_w:
        Actual valid image area.

    Notes
    -----
    Normally patch is already 224x224.

    Padding is only needed if the entire input image itself is
    smaller than patch_size.
    """

    ph, pw = patch_np.shape

    valid_h = ph
    valid_w = pw

    # --------------------------------------------------------
    # If original image itself is smaller than patch_size
    # --------------------------------------------------------
    if ph != patch_size or pw != patch_size:

        patch_canvas = np.zeros(
            (patch_size, patch_size),
            dtype=patch_np.dtype
        )

        patch_canvas[
            :ph,
            :pw
        ] = patch_np

        patch_np = patch_canvas

    # --------------------------------------------------------
    # numpy -> torch
    #
    # uint8 0~255
    # ->
    # float32 0~1
    # --------------------------------------------------------

    patch_tensor = torch.from_numpy(
        patch_np
    ).float()

    patch_tensor = patch_tensor / 255.0

    # [H, W]
    # ->
    # [1, H, W]
    # ->
    # [1, 1, H, W]

    patch_tensor = (
        patch_tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    return (
        patch_tensor,
        valid_h,
        valid_w
    )


# ============================================================
# Save probability map
# ============================================================

def save_probability_map(
    prob_map,
    save_path
):
    """
    Save a probability map in [0, 1] as uint8 PNG.

    IMPORTANT:
    Do NOT normalize using pm / pm.max(),
    because that destroys the actual probability scale.
    """

    prob_uint8 = (
        np.clip(
            prob_map,
            0.0,
            1.0
        ) * 255.0
    ).astype(np.uint8)

    Image.fromarray(
        prob_uint8
    ).save(
        save_path
    )


# ============================================================
# Single-image tiled inference
# ============================================================

def inference_single_image(
    image_path,
    model,
    save_dir,
    patch_size=224,
    stride=224,
    num_classes=3,
    device="cuda",
    death_prob_is_logits=False
):
    """
    Whole-image tiled inference.

    Pipeline
    --------
    Original image
        ↓
    generate_patch_coords
        ↓
    224x224 patches
        ↓
    DoANet inference
        ↓
    segmentation probability
    + death probability
        ↓
    accumulate according to coordinates
        ↓
    divide by count_map
        ↓
    full-resolution probability maps
        ↓
    argmax
        ↓
    final full-resolution segmentation
    """

    model.eval()

    # ========================================================
    # 1. Load original image
    # ========================================================

    image_pil = Image.open(
        image_path
    ).convert("L")

    image_np = np.array(
        image_pil
    )

    H, W = image_np.shape

    case_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    logging.info(
        f"Image: {case_name}"
    )

    logging.info(
        f"Original size: H={H}, W={W}"
    )

    # ========================================================
    # 2. Generate patch coordinates
    # ========================================================

    coords = generate_patch_coords(
        image_shape=image_np.shape,
        patch_size=patch_size,
        stride=stride
    )

    logging.info(
        f"Patch size: {patch_size}"
    )

    logging.info(
        f"Stride: {stride}"
    )

    logging.info(
        f"Total patches: {len(coords)}"
    )

    # ========================================================
    # 3. Whole-image accumulation canvas
    # ========================================================

    # segmentation probability sum
    #
    # [C, H, W]

    prob_sum = np.zeros(
        (
            num_classes,
            H,
            W
        ),
        dtype=np.float32
    )

    # death probability sum
    #
    # [H, W]

    death_sum = np.zeros(
        (
            H,
            W
        ),
        dtype=np.float32
    )

    # record how many patches cover every pixel
    #
    # [H, W]

    count_map = np.zeros(
        (
            H,
            W
        ),
        dtype=np.float32
    )

    # ========================================================
    # 4. Patch inference
    # ========================================================

    with torch.no_grad():

        for patch_idx, coord in enumerate(
            tqdm(
                coords,
                desc=f"Inference {case_name}"
            )
        ):

            x1, y1, x2, y2 = coord

            # ------------------------------------------------
            # Extract patch
            # ------------------------------------------------

            patch_np = image_np[
                y1:min(y2, H),
                x1:min(x2, W)
            ]

            # ------------------------------------------------
            # Convert patch into tensor
            # ------------------------------------------------

            (
                patch_tensor,
                valid_h,
                valid_w
            ) = patch_to_tensor(
                patch_np,
                patch_size=patch_size,
                device=device
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------
            #
            # Expected:
            #
            # logits:
            # [1, C, 224, 224]
            #
            # death_prob:
            # [1, 1, 224, 224]
            #

            logits, death_prob = model(
                patch_tensor
            )

            # ------------------------------------------------
            # segmentation logits -> probabilities
            # ------------------------------------------------

            probs = torch.softmax(
                logits,
                dim=1
            )

            # [1, C, H, W]
            # ->
            # [C, H, W]

            probs_np = (
                probs[0]
                .detach()
                .cpu()
                .numpy()
            )

            # only keep valid area
            #
            # Normally:
            # valid_h = valid_w = 224
            #
            # Only relevant when original image < 224

            probs_valid = probs_np[
                :,
                :valid_h,
                :valid_w
            ]

            # ------------------------------------------------
            # Stitch segmentation probabilities
            # ------------------------------------------------

            prob_sum[
                :,
                y1:y1 + valid_h,
                x1:x1 + valid_w
            ] += probs_valid

            # ------------------------------------------------
            # Death probability
            # ------------------------------------------------

            if death_prob is not None:

                # If model returns logits rather than sigmoid probability,
                # convert them here.
                #
                # Set:
                #
                # --death_prob_is_logits
                #
                # when running script.

                if death_prob_is_logits:

                    death_prob = torch.sigmoid(
                        death_prob
                    )

                # Expected:
                #
                # [1, 1, 224, 224]

                if death_prob.ndim == 4:

                    death_np = (
                        death_prob[
                            0,
                            0
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                elif death_prob.ndim == 3:

                    death_np = (
                        death_prob[
                            0
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                else:

                    raise ValueError(
                        "Unexpected death_prob shape: "
                        f"{death_prob.shape}"
                    )

                death_valid = death_np[
                    :valid_h,
                    :valid_w
                ]

                death_sum[
                    y1:y1 + valid_h,
                    x1:x1 + valid_w
                ] += death_valid

            # ------------------------------------------------
            # Update coverage map
            # ------------------------------------------------

            count_map[
                y1:y1 + valid_h,
                x1:x1 + valid_w
            ] += 1.0

    # ========================================================
    # 5. Check coverage
    # ========================================================

    uncovered_pixels = np.sum(
        count_map == 0
    )

    if uncovered_pixels > 0:

        raise RuntimeError(
            f"{uncovered_pixels} pixels were not covered "
            "by any patch."
        )

    logging.info(
        "All pixels covered."
    )

    logging.info(
        f"Coverage count range: "
        f"{count_map.min()} ~ {count_map.max()}"
    )

    # ========================================================
    # 6. Average overlapping predictions
    # ========================================================

    # count_map:
    #
    # [H, W]
    #
    # probability:
    #
    # [C, H, W]
    #
    # therefore expand:
    #
    # [1, H, W]

    prob_avg = (
        prob_sum /
        count_map[
            None,
            :,
            :
        ]
    )

    death_avg = (
        death_sum /
        count_map
    )

    # ========================================================
    # 7. Final segmentation
    # ========================================================

    final_pred = np.argmax(
        prob_avg,
        axis=0
    ).astype(np.uint8)

    logging.info(
        f"Final prediction shape: "
        f"{final_pred.shape}"
    )

    # ========================================================
    # 8. Create output folders
    # ========================================================

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    input_dir = os.path.join(
        save_dir,
        "input"
    )

    pred_dir = os.path.join(
        save_dir,
        "pred"
    )

    prob_dir = os.path.join(
        save_dir,
        "prob"
    )

    death_dir = os.path.join(
        save_dir,
        "death_prob"
    )

    coverage_dir = os.path.join(
        save_dir,
        "coverage"
    )

    os.makedirs(
        input_dir,
        exist_ok=True
    )

    os.makedirs(
        pred_dir,
        exist_ok=True
    )

    os.makedirs(
        prob_dir,
        exist_ok=True
    )

    os.makedirs(
        death_dir,
        exist_ok=True
    )

    os.makedirs(
        coverage_dir,
        exist_ok=True
    )

    # ========================================================
    # 9. Save original input
    # ========================================================

    input_save_path = os.path.join(
        input_dir,
        f"{case_name}_input.png"
    )

    Image.fromarray(
        image_np
    ).save(
        input_save_path
    )

    # ========================================================
    # 10. Save class probability maps
    # ========================================================

    for cls_idx in range(
        num_classes
    ):

        class_prob = prob_avg[
            cls_idx
        ]

        prob_save_path = os.path.join(
            prob_dir,
            f"{case_name}_prob_class{cls_idx}.png"
        )

        save_probability_map(
            class_prob,
            prob_save_path
        )

    # ========================================================
    # 11. Save segmentation
    # ========================================================
    #
    # For 3 classes:
    #
    # class 0 -> 0
    # class 1 -> 127
    # class 2 -> 254
    #

    if num_classes > 1:

        vis = (
            final_pred.astype(np.float32)
            /
            float(num_classes - 1)
            *
            255.0
        ).astype(np.uint8)

    else:

        vis = (
            final_pred * 255
        ).astype(np.uint8)

    pred_save_path = os.path.join(
        pred_dir,
        f"{case_name}_pred.png"
    )

    Image.fromarray(
        vis
    ).save(
        pred_save_path
    )

    # --------------------------------------------------------
    # Also save raw class index map
    #
    # pixel values:
    #
    # 0 / 1 / 2
    # --------------------------------------------------------

    raw_pred_save_path = os.path.join(
        pred_dir,
        f"{case_name}_pred_raw.png"
    )

    Image.fromarray(
        final_pred
    ).save(
        raw_pred_save_path
    )

    # ========================================================
    # 12. Save death probability
    # ========================================================

    death_save_path = os.path.join(
        death_dir,
        f"{case_name}_death_prob.png"
    )

    save_probability_map(
        death_avg,
        death_save_path
    )

    # ========================================================
    # 13. Save count map
    # ========================================================
    #
    # Mainly for checking overlap/stitching.
    #

    count_max = count_map.max()

    count_vis = (
        count_map /
        max(
            count_max,
            1
        )
        *
        255
    ).astype(np.uint8)

    coverage_save_path = os.path.join(
        coverage_dir,
        f"{case_name}_count_map.png"
    )

    Image.fromarray(
        count_vis
    ).save(
        coverage_save_path
    )

    # ========================================================
    # 14. Save patch coordinate record
    # ========================================================

    coord_save_path = os.path.join(
        save_dir,
        f"{case_name}_patch_coords.txt"
    )

    with open(
        coord_save_path,
        "w"
    ) as f:

        f.write(
            "patch_idx,x1,y1,x2,y2\n"
        )

        for idx, (
            x1,
            y1,
            x2,
            y2
        ) in enumerate(coords):

            f.write(
                f"{idx},"
                f"{x1},"
                f"{y1},"
                f"{x2},"
                f"{y2}\n"
            )

    logging.info(
        f"Inference completed: {case_name}"
    )

    logging.info(
        f"Saved to: {save_dir}"
    )

    return {
        "pred": final_pred,
        "prob": prob_avg,
        "death_prob": death_avg,
        "coords": coords,
        "count_map": count_map
    }


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="DoANet tiled single-image inference"
)


# ------------------------------------------------------------
# Input image
# ------------------------------------------------------------

parser.add_argument(
    "--image_path",
    type=str,
    required=True,
    help="Path to one full-resolution image"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Model/checkpoint directory"
)

parser.add_argument(
    "--test_save_dir",
    type=str,
    default="./predictions",
    help="Output directory for prediction"
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

parser.add_argument(
    "--num_classes",
    type=int,
    default=3,
    help="Number of segmentation classes"
)

parser.add_argument(
    "--img_size",
    type=int,
    default=224,
    help="Patch size"
)

parser.add_argument(
    "--stride",
    type=int,
    default=224,
    help="Patch stride"
)


# ------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help=(
        "Checkpoint path. "
        "If not given, output_dir/inference.pth will be used."
    )
)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

parser.add_argument(
    "--deterministic",
    type=int,
    default=1
)

parser.add_argument(
    "--seed",
    type=int,
    default=1234
)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

parser.add_argument(
    "--cfg",
    type=str,
    required=True,
    metavar="FILE",
    help="Path to config file"
)

parser.add_argument(
    "--opts",
    help="Modify config options by adding KEY VALUE pairs",
    default=None,
    nargs="+"
)

parser.add_argument(
    "--zip",
    action="store_true",
    help="Use zipped dataset"
)

parser.add_argument(
    "--cache-mode",
    type=str,
    default="part",
    choices=[
        "no",
        "full",
        "part"
    ]
)

parser.add_argument(
    "--resume",
    help="Resume from checkpoint"
)

parser.add_argument(
    "--accumulation-steps",
    type=int
)

parser.add_argument(
    "--use-checkpoint",
    action="store_true"
)

parser.add_argument(
    "--amp-opt-level",
    type=str,
    default="O1",
    choices=[
        "O0",
        "O1",
        "O2"
    ]
)

parser.add_argument(
    "--tag"
)

parser.add_argument(
    "--eval",
    action="store_true"
)

parser.add_argument(
    "--throughput",
    action="store_true"
)


# ------------------------------------------------------------
# Original config compatibility
# ------------------------------------------------------------

parser.add_argument(
    "--batch_size",
    type=int,
    default=1
)

parser.add_argument(
    "--max_iterations",
    type=int,
    default=30000
)

parser.add_argument(
    "--max_epochs",
    type=int,
    default=150
)

parser.add_argument(
    "--base_lr",
    type=float,
    default=0.01
)

parser.add_argument(
    "--dataset",
    type=str,
    default="Organoids"
)

parser.add_argument(
    "--volume_path",
    type=str,
    default=""
)

parser.add_argument(
    "--list_dir",
    type=str,
    default=""
)


# ------------------------------------------------------------
# Death probability output type
# ------------------------------------------------------------

parser.add_argument(
    "--death_prob_is_logits",
    action="store_true",
    help=(
        "Use this flag if DoANet returns raw logits "
        "for the death probability branch."
    )
)


args = parser.parse_args()

config = get_config(args)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # CUDA
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = "cuda"

    # ========================================================
    # Deterministic settings
    # ========================================================

    if not args.deterministic:

        cudnn.benchmark = True
        cudnn.deterministic = False

    else:

        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    # ========================================================
    # Check input image
    # ========================================================

    if not os.path.exists(
        args.image_path
    ):

        raise FileNotFoundError(
            f"Image not found: {args.image_path}"
        )

    # ========================================================
    # Create model
    # ========================================================

    args.is_pretrain = True

    net = DoANet(
        config,
        img_size=args.img_size,
        num_classes=args.num_classes
    ).to(device)

    # ========================================================
    # Find checkpoint
    # ========================================================

    if args.checkpoint is not None:

        snapshot = args.checkpoint

    else:

        snapshot = os.path.join(
            args.output_dir,
            "inference.pth"
        )

    if not os.path.exists(
        snapshot
    ):

        raise FileNotFoundError(
            f"Checkpoint not found: {snapshot}"
        )

    print(
        f"Loading checkpoint: {snapshot}"
    )

    checkpoint = torch.load(
        snapshot,
        map_location="cpu"
    )

    # ========================================================
    # Load model
    # ========================================================
    #
    # Keep behavior compatible with your original script.
    #

    if isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        checkpoint = checkpoint[
            "state_dict"
        ]

    msg = net.load_state_dict(
        checkpoint,
        strict=False
    )

    print(
        "\nMissing keys:"
    )

    if len(msg.missing_keys) == 0:

        print(
            "  None"
        )

    else:

        for key in msg.missing_keys:

            print(
                "  ",
                key
            )

    print(
        "\nUnexpected keys:"
    )

    if len(msg.unexpected_keys) == 0:

        print(
            "  None"
        )

    else:

        for key in msg.unexpected_keys:

            print(
                "  ",
                key
            )

    # ========================================================
    # Logging
    # ========================================================

    snapshot_name = os.path.basename(
        snapshot
    )

    log_folder = "./test_log"

    os.makedirs(
        log_folder,
        exist_ok=True
    )

    log_path = os.path.join(
        log_folder,
        f"{snapshot_name}_single_image.txt"
    )

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format=(
            "[%(asctime)s.%(msecs)03d] "
            "%(message)s"
        ),
        datefmt="%H:%M:%S"
    )

    logging.getLogger().addHandler(
        logging.StreamHandler(
            sys.stdout
        )
    )

    logging.info(
        str(args)
    )

    logging.info(
        f"Checkpoint: {snapshot}"
    )

    # ========================================================
    # Output directory
    # ========================================================

    image_name = Path(
        args.image_path
    ).stem

    test_save_path = os.path.join(
        args.test_save_dir,
        image_name
    )

    os.makedirs(
        test_save_path,
        exist_ok=True
    )

    logging.info(
        f"Save directory: {test_save_path}"
    )

    # ========================================================
    # Inference
    # ========================================================

    result = inference_single_image(
        image_path=args.image_path,
        model=net,
        save_dir=test_save_path,
        patch_size=args.img_size,
        stride=args.stride,
        num_classes=args.num_classes,
        device=device,
        death_prob_is_logits=args.death_prob_is_logits
    )

    logging.info(
        "Inference Finished!"
    )