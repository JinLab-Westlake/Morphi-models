import os
import cv2
import numpy as np
import pandas as pd

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_image_any(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img

def fit_to_roi(tile, roi_w, roi_h):
    h, w = tile.shape[:2]
    if (w, h) == (roi_w, roi_h):
        return tile
    interp = cv2.INTER_NEAREST if tile.dtype != np.float32 else cv2.INTER_LINEAR
    return cv2.resize(tile, (roi_w, roi_h), interpolation=interp)

def paste_tile(canvas, tile, x0, y0, x1, y1):
    H, W = canvas.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c <= x0c or y1c <= y0c:
        return

    roi_w, roi_h = x1 - x0, y1 - y0
    tile = fit_to_roi(tile, roi_w, roi_h)

    tx0 = x0c - x0
    ty0 = y0c - y0
    tx1 = tx0 + (x1c - x0c)
    ty1 = ty0 + (y1c - y0c)

    canvas[y0c:y1c, x0c:x1c] = tile[ty0:ty1, tx0:tx1]

def reconstruct_all(txt_path, pred_folder, out_dir,
                    full_size=(1080, 1080),
                    pred_suffix="_pred_unpad.png",
                    canvas_channels=None):
    """
    canvas_channels:
        None -> Automatically decide based on the first tile (grayscale/color/alpha)
        1/3/4 -> Force the number of output channels
    """
    ensure_dir(out_dir)

    df = pd.read_csv(txt_path)
    needed = ["base", "x0", "y0", "x1", "y1", "img_out"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"txt don't have: {c}，current line: {list(df.columns)}")

    H, W = full_size[1], full_size[0]

    # group
    for base, g in df.groupby("base", sort=False):
        g = g.sort_values("idx") if "idx" in g.columns else g

        canvas = None
        decided_channels = canvas_channels

        missing = 0
        pasted = 0

        for _, row in g.iterrows():
            x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
            img_out = str(row["img_out"])  # such as r01c01f01_crop1.png

            pred_name = img_out + pred_suffix  # r01c01f01_crop1.png_pred_unpad.png
            pred_path = os.path.join(pred_folder, pred_name)
            if not os.path.exists(pred_path):
                missing += 1
                continue

            tile = read_image_any(pred_path)

            # initialize canvas
            if canvas is None:
                if decided_channels is None:
                    if tile.ndim == 2:
                        decided_channels = 1
                    else:
                        decided_channels = tile.shape[2]  # 3 or 4
                if decided_channels == 1:
                    canvas = np.zeros((H, W), dtype=tile.dtype)
                else:
                    canvas = np.zeros((H, W, decided_channels), dtype=tile.dtype)

            # If the tile channel does not align with the canvas, do a simple alignment
            if decided_channels == 1:
                if tile.ndim == 3:
                    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if tile.shape[2] >= 3 else tile[:, :, 0]
            else:
                if tile.ndim == 2:
                    tile = np.repeat(tile[:, :, None], decided_channels, axis=2)
                elif tile.shape[2] != decided_channels:
                    if tile.shape[2] == 4 and decided_channels == 3:
                        tile = tile[:, :, :3]
                    elif tile.shape[2] == 3 and decided_channels == 4:
                        alpha = np.full(tile.shape[:2] + (1,), 255, dtype=tile.dtype)
                        tile = np.concatenate([tile, alpha], axis=2)
                    else:
                        tile = tile[:, :, :decided_channels]

            paste_tile(canvas, tile, x0, y0, x1, y1)
            pasted += 1

        if canvas is None:
            print(f"[WARN] {base}: No tile found")
            continue

        out_path = os.path.join(out_dir, f"{base}_reconstructed.png")
        cv2.imwrite(out_path, canvas)
        print(f"[OK] {base}: pasted={pasted}, missing={missing}, saved -> {out_path}")

if __name__ == "__main__":
    txt_path = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20251216/20251213_NCIH460_NO1_P01-P04__2025-12-13T20_58_15-Measurement_1/Combined/patches/positions.txt"
    pred_folder = "/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/model_out20260204_padding_enhanced/20260208/pred/NCIH460_p1"
    out_dir = "/storage/jinxinLab/wangyimin/Cell_seg/Swin-Unet-main/model_out20260204_padding_enhanced/20260208/recombined/NCIH460_p1"

    reconstruct_all(
        txt_path=txt_path,
        pred_folder=pred_folder,
        out_dir=out_dir,
        full_size=(1080, 1080),
        pred_suffix="_pred_unpad.png",
        canvas_channels=None,
    )
