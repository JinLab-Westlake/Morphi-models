#!/usr/bin/env python3
"""
Recursive batch: Flat-field illumination correction + percentile normalization [0,1] + background robust Z-score + (optional)QC panel
Dependencies: numpy, matplotlib, scikit-image (skimage)
"""

"""
usage:
python flatfield_batch.py \
  --in_dir /data/bf_imgs \
  --out_dir /data/bf_out \
  --sigma 50 --p_lo 1 --p_hi 99 \
  --bg_method otsu \
  --save_modes ff,pct,z,qc \
  --exts .tif,.tiff,.png,.jpg \
  --workers 8
"""

from pathlib import Path
import argparse, traceback
import numpy as np
from skimage import io
from skimage.filters import gaussian, threshold_otsu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional


EPS = 1e-6

# Base handler function
def read_gray(path: Path) -> np.ndarray:
    img = io.imread(str(path))
    if img.ndim == 3:  # RGB -> gray
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    return img.astype(np.float32)

def flatfield_correct(img: np.ndarray, sigma: float):
    F = gaussian(img, sigma=sigma, preserve_range=True)
    out = img / (F + EPS) * np.median(F)
    return out, F

def percentile_rescale(img: np.ndarray, p_lo: float, p_hi: float):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi <= lo:
        x = np.zeros_like(img, dtype=np.float32)
    else:
        x = np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)
    return x, (float(lo), float(hi))

def estimate_bg_mask(img_ff: np.ndarray, method="otsu", pct=30.0):
    if method == "otsu":
        try:
            th = threshold_otsu(img_ff)
        except ValueError:
            th = float(np.median(img_ff))
        bg = img_ff <= th
    else:
        k = float(np.percentile(img_ff, pct))
        bg = img_ff <= k
    return bg


def robust_zscore_with_mask(img: np.ndarray, mask: Optional[np.ndarray] = None):
    """
    Compute a robust z-score on pixels with mask=True: Z = (x-median)/(1.4826 * MAD)
    If mask=None, the full image is used.
    """
    if mask is not None:
        vals = img[mask]
    else:
        vals = img.ravel()

    if vals.size < 10:
        med = float(np.median(img))
        mad = float(np.median(np.abs(img - med))) + EPS
    else:
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + EPS

    z = (img - med) / (1.4826 * mad)
    return z.astype(np.float32), (med, mad)



def save_qc_panel(img_raw, F, img_ff, img_pct, img_z, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])

    ax1 = fig.add_subplot(gs[0,0]); ax1.imshow(img_raw, cmap="gray"); ax1.set_title("Raw"); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0,1]); ax2.imshow(F, cmap="gray"); ax2.set_title("Flat-field F"); ax2.axis("off")
    ax3 = fig.add_subplot(gs[0,2]); ax3.imshow(img_ff, cmap="gray"); ax3.set_title("Flat-field corrected"); ax3.axis("off")

    ax4 = fig.add_subplot(gs[1,0]); ax4.imshow(img_pct, cmap="gray", vmin=0, vmax=1); ax4.set_title("Percentile [0,1]"); ax4.axis("off")
    ax5 = fig.add_subplot(gs[1,1]); im = ax5.imshow(img_z, cmap="coolwarm"); ax5.set_title("Background robust z-score"); ax5.axis("off")
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1,2])
    ax6.hist(img_raw.ravel(), bins=256, alpha=0.5, label="raw")
    ax6.hist(img_ff.ravel(),  bins=256, alpha=0.5, label="flatfield")
    ax6.hist(img_pct.ravel(), bins=256, alpha=0.5, label="rescaled[0,1]")
    ax6.set_title("Intensity histograms"); ax6.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=180)
    plt.close()

# visualization
def z_to_uint8(img_z: np.ndarray, clip=3.0) -> np.ndarray:
    zc = np.clip(img_z, -clip, clip)
    # project [-clip,clip] -> [0,1]
    vis = (zc + clip) / (2*clip)
    return (vis * 255.0 + 0.5).astype(np.uint8)

def process_one(in_path: Path, out_root: Path, rel_root: Path,
                sigma: float, p_lo: float, p_hi: float,
                bg_method: str, bg_pct: float,
                save_modes: set, z_vis_clip: float,
                pct_mask_mode: str = "noborder",
                ignore_border_px: int = 0):
    try:
        img_raw = read_gray(in_path)
        img_ff, F = flatfield_correct(img_raw, sigma=sigma)

        # Estimate the background mask first (for subsequent nonbg mode)
        bg_mask = estimate_bg_mask(img_ff, method=bg_method, pct=bg_pct)

        # Construct the "participate in quantile statistics" mask and do the [0,1] percentile stretch
        pct_mask = build_pct_mask(img_ff, mode=pct_mask_mode,
                                  ignore_border_px=ignore_border_px,
                                  bg_mask=bg_mask)
        img_pct, (lo, hi) = percentile_rescale_with_mask(img_ff, p_lo, p_hi, mask=pct_mask)

        # Background robust z-score
        # img_z, (med_bg, mad_bg) = robust_bg_zscore(img_ff, bg_mask)
        img_z,(med_bg, mad_bg) =robust_zscore_with_mask(img_ff,mask = pct_mask)
        

        # Output path and save
        rel  = in_path.relative_to(rel_root)
        stem = rel.with_suffix("")
        out_dir = (out_root / stem).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if "ff" in save_modes:
            io.imsave(str(out_root / f"{stem}.ff.tif"), img_ff.astype(np.float32))
        if "pct" in save_modes:
            io.imsave(str(out_root / f"{stem}.pct.png"), (img_pct * 255.0 + 0.5).astype(np.uint8))
        if "z" in save_modes:
            io.imsave(str(out_root / f"{stem}.z.png"), z_to_uint8(img_z, clip=z_vis_clip))
            np.save(str(out_root / f"{stem}.z.npy"), img_z.astype(np.float32))
        if "bgmask" in save_modes:
            io.imsave(str(out_root / f"{stem}.bgmask.png"), (bg_mask.astype(np.uint8) * 255))
        if "qc" in save_modes:
            save_qc_panel(img_raw, F, img_ff, img_pct, img_z, out_root / f"{stem}.qc.png")

        return (str(in_path), True,
                f"p[{p_lo},{p_hi}] lo/hi={lo:.3f}/{hi:.3f}, bg_med={med_bg:.3f}, MAD={mad_bg:.3f}")
    except Exception as e:
        return (str(in_path), False, f"{e}\n{traceback.format_exc()}")



def build_pct_mask(img: np.ndarray,
                   mode: str = "none",
                   ignore_border_px: int = 0,
                   bg_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Returns a Boolean mask representing "pixels participating in quantile statistics".
    mode:
    - 'none' : No mask is used
    - 'noborder' : Ignores surrounding ignore_border_px pixels
    - 'nonbg' : uses ~bg_mask (non-background) as the statistics field; bg_mask is required first
    - 'center' : Use only the center area (minus the surrounding ignore_border_px)
    """
    H, W = img.shape[:2]
    if mode == "none":
        return None

    mask = np.ones((H, W), dtype=bool)

    if mode in ("noborder", "center") and ignore_border_px > 0:
        p = int(ignore_border_px)
        mask[:p, :] = False; mask[-p:, :] = False
        mask[:, :p] = False; mask[:, -p:] = False

    if mode == "nonbg":
        if bg_mask is None:
            # No background provided, degenerates to noborder
            return build_pct_mask(img, mode="noborder", ignore_border_px=ignore_border_px, bg_mask=None)
        mask = ~bg_mask.astype(bool)

    return mask

def percentile_rescale_with_mask(img: np.ndarray, p_lo: float, p_hi: float,
                                 mask: Optional[np.ndarray] = None):
    """p_lo/p_hi is computed only on pixels where mask=True,
    and then the whole image is linearly mapped to [0,1]."""

    if mask is None:
        lo, hi = np.percentile(img, [p_lo, p_hi])
    else:
        vals = img[mask]
        if vals.size < 10:
            lo, hi = np.percentile(img, [p_lo, p_hi])
        else:
            lo, hi = np.percentile(vals, [p_lo, p_hi])
    if hi <= lo:
        x = np.zeros_like(img, dtype=np.float32)
    else:
        x = np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)
    return x, (float(lo), float(hi))



# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Input")
    ap.add_argument("--out_dir", required=True, help="Output")
    ap.add_argument("--sigma", type=float, default=50.0, help="Gaussian σ in pixels estimates the illumination field F")
    ap.add_argument("--p_lo", type=float, default=1.0, help="The percentile stretch lower bound")
    ap.add_argument("--p_hi", type=float, default=99.0, help="The percentile stretch upper bound")
    ap.add_argument("--bg_method", default="otsu", choices=["otsu","percentile"], help="Background estimation method")
    ap.add_argument("--bg_pct", type=float, default=30.0, help="bg_method=percentile, the lowest percentage")
    ap.add_argument("--save_modes", default="ff,pct,z,qc", help="Comma-separated:ff,pct,z,bgmask,qc")
    ap.add_argument("--exts", default=".tif,.tiff,.png,.jpg,.jpeg", help="Suffix matched recursively")
    ap.add_argument("--workers", type=int, default=16, help="Number of parallel processes")
    ap.add_argument("--z_vis_clip", type=float, default=3.0, help="z-score visual clipping interval [-c,c]")
    ap.add_argument("--ignore_border_px", type=int, default=50,
                help="The surrounding pixel widths (pixels) are ignored when calculating the quantiles")
    ap.add_argument("--pct_mask", default="noborder",
                choices=["none","noborder","nonbg","center"],
                help="Quantile field: none= full image; noborder= Ignore borders; nonbg= non-background; center= Center region only")

    args = ap.parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    save_modes = {m.strip().lower() for m in args.save_modes.split(",") if m.strip()}

    # Collect file
    files = [p for p in in_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    print(f"[INFO] Found {len(files)} images under {in_root}")

    # Parallel processing
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [
            ex.submit(
                process_one, p, out_root, in_root,
                args.sigma, args.p_lo, args.p_hi,
                args.bg_method, args.bg_pct,
                save_modes, args.z_vis_clip,
                args.pct_mask, args.ignore_border_px
            ) for p in files
        ]

        for i, fut in enumerate(as_completed(futs), 1):
            in_path, success, msg = fut.result()
            prefix = "OK " if success else "ERR"
            print(f"[{i:>5d}/{len(files)}] {prefix} - {in_path}\n      {msg}")
            ok += int(success); fail += int(not success)

    print(f"[DONE] success={ok}, failed={fail}, out_dir={out_root}")

if __name__ == "__main__":
    main()
