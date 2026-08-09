"""
Run per-object feature extraction using a pairs CSV produced earlier.

Input CSV (from pairing step) must have columns:
  drugname, plate, token, row, col, folder_name, brightfield_path, mask_path

Usage:
  python run_feature_extract.py \
    --pairs_csv /path/to/pairs.csv \
    --out_csv   /path/to/features.csv \
    [--out_xlsx /path/to/features.xlsx] \
    [--pixel_size 0.5] \
    [--glcm_dist 1 2] \
    [--glcm_angles 0 0.78539816339 1.57079632679 2.35619449019] \
    [--glcm_levels 64]
"""

import argparse
import math
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io as skio
from skimage.color import rgb2gray
from skimage.measure import regionprops, label as cc_label, perimeter_crofton, find_contours
from skimage.morphology import convex_hull_image
from skimage.segmentation import find_boundaries
from skimage.feature import graycomatrix, graycoprops

from pathlib import Path
from skimage.morphology import remove_small_holes
import cv2
import re


def get_fov_key(x):
    """
    Extract keys such as r01c02f01 from the filename or path.
    """
    s = Path(str(x)).name
    m = re.search(r"(r\d{2}c\d{2}f\d{2})", s)
    if m:
        return m.group(1)
    return Path(s).stem


# -------------------- Helpers for geometry & texture --------------------
def _perimeter_subpixel(mask, directions=4):
    return float(perimeter_crofton(mask.astype(bool), directions=directions))

def _circularity(area, perimeter):
    return float(4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else np.nan

def _convex_hull_and_perim(mask):
    hull = convex_hull_image(mask)
    perim = _perimeter_subpixel(hull)
    return hull, float(perim)

def _feret_max_min_from_hull(hull_mask):
    contours = find_contours(hull_mask.astype(float), 0.5)
    if not contours:
        return np.nan, np.nan
    poly = contours[0]  # (n,2) -> (y,x)
    P = np.stack([poly[:,1], poly[:,0]], axis=1).astype(np.float64)  # (x,y)
    n = len(P)
    if n < 3:
        return np.nan, np.nan
    diffs = P[None,:,:] - P[:,None,:]
    d2 = (diffs ** 2).sum(axis=2)
    feret_max = float(np.sqrt(d2.max()))
    feret_min = np.inf
    for i in range(n):
        j = (i + 1) % n
        edge = P[j] - P[i]
        if np.allclose(edge, 0): 
            continue
        nrm = np.array([-edge[1], edge[0]], dtype=np.float64)
        norm = np.linalg.norm(nrm)
        if norm == 0:
            continue
        nrm /= norm
        proj = P @ nrm
        width = proj.max() - proj.min()
        if width < feret_min:
            feret_min = width
    return float(feret_max), (float(feret_min) if np.isfinite(feret_min) else np.nan)

def _fractal_dim_boxcount(boundary_mask):
    bw = boundary_mask.astype(bool)
    if not bw.any():
        return np.nan
    h, w = bw.shape
    sizes, counts = [], []
    max_k = int(np.floor(np.log2(min(h, w))))
    for k in range(max_k, 0, -1):
        box = 2 ** k
        nH = (h + box - 1) // box
        nW = (w + box - 1) // box
        c = 0
        for i in range(nH):
            for j in range(nW):
                sl = bw[i*box:(i+1)*box, j*box:(j+1)*box]
                if sl.any():
                    c += 1
        if c > 0:
            sizes.append(box)
            counts.append(c)
    if len(counts) < 2:
        return np.nan
    x = np.log(1.0 / np.asarray(sizes))
    y = np.log(np.asarray(counts))
    A = np.vstack([x, np.ones_like(x)]).T
    D, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(D)

def _radial_std(boundary_mask, centroid_xy):
    ys, xs = np.nonzero(boundary_mask)
    if len(xs) == 0:
        return np.nan
    cx, cy = centroid_xy
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return float(r.std())

def _quantize_to_levels(img, levels=64, mask=None):
    img = img.astype(np.float32)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax <= vmin:
        q = np.zeros_like(img, dtype=np.uint8)
        return q, 1
    q = np.floor((img - vmin) / (vmax - vmin + 1e-12) * (levels - 1) + 0.5).astype(np.uint8)
    if mask is not None:
        mean_bin = int(np.round(q[mask].mean())) if mask.any() else 0
        q[~mask] = mean_bin
    return q, levels

def _glcm_all_props(gray_q, mask, distances=(1,), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4), levels=64):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {k: np.nan for k in ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]}
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    roi = gray_q[y0:y1+1, x0:x1+1].copy()
    mroi = mask[y0:y1+1, x0:x1+1]
    mean_bin = int(np.round(roi[mroi].mean())) if mroi.any() else 0
    roi[~mroi] = mean_bin
    P = graycomatrix(roi, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    out = {}
    for name in ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]:
        out[name] = float(graycoprops(P, name).mean())
    return out

def _entropy_bits_from_vals(vals_u8):
    if vals_u8.size == 0:
        return np.nan
    hist, _ = np.histogram(vals_u8, bins=256, range=(0,255), density=True)
    p = hist[hist > 0]
    return float(-(p * np.log2(p)).sum())




def extract_features(instance_label, brightfield=None, pixel_size=None,
                     glcm_distances=(1,),
                     glcm_angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                     glcm_levels=64):
    """
    instance_label:
        0 = background
        1,2,3,... = object instance id
    """

    lab = instance_label.astype(np.int32)

    rows = []

    for rp in regionprops(lab):
        lbl = int(rp.label)
        region_mask = (lab == lbl)

        # ====== 形状特征 ======
        area_px = float(rp.area)
        perim_px = _perimeter_subpixel(region_mask, directions=4)
        circ = _circularity(area_px, perim_px)

        major = float(rp.major_axis_length) if rp.major_axis_length is not None else np.nan
        minor = float(rp.minor_axis_length) if rp.minor_axis_length is not None else np.nan
        aspect = float(major / minor) if (minor and minor > 0) else np.nan

        hull, ch_perim_px = _convex_hull_and_perim(region_mask)
        ch_area_px = float(hull.sum())
        solidity = float(area_px / ch_area_px) if ch_area_px > 0 else np.nan
        irregularity = float(perim_px / ch_perim_px) if ch_perim_px > 0 else np.nan

        feret_max_px, feret_min_px = _feret_max_min_from_hull(hull)

        boundary = find_boundaries(region_mask, mode='outer')
        fd = _fractal_dim_boxcount(boundary)
        rad_std = _radial_std(boundary, centroid_xy=(rp.centroid[1], rp.centroid[0]))

        comp_count = int(cc_label(region_mask, connectivity=2).max())

        # ====== Texture/brightness ======
        if brightfield is not None:
            bf = brightfield.astype(np.float32)
            vals = bf[region_mask]

            bf_mean = float(vals.mean()) if vals.size else np.nan
            bf_var = float(vals.var()) if vals.size else np.nan

            q_img, L = _quantize_to_levels(bf, levels=glcm_levels, mask=region_mask)
            glcm = _glcm_all_props(
                q_img,
                region_mask,
                distances=glcm_distances,
                angles=glcm_angles,
                levels=L
            )

            vmax = bf.max() if bf.size else 1.0
            vals_u8 = (
                np.clip(vals, 0, None) / (vmax + 1e-12) * 255
            ).astype(np.uint8) if vals.size else np.array([], np.uint8)

            entropy = _entropy_bits_from_vals(vals_u8)
        else:
            bf_mean = bf_var = entropy = np.nan
            glcm = {
                k: np.nan for k in [
                    "contrast",
                    "dissimilarity",
                    "homogeneity",
                    "ASM",
                    "energy",
                    "correlation"
                ]
            }

        # ====== Conversion of units ======
        if pixel_size is not None:
            px = float(pixel_size)
            area = area_px * (px ** 2)
            perim = perim_px * px
            feret_max = feret_max_px * px
            feret_min = feret_min_px * px
            unit_area = "µm^2"
            unit_len = "µm"
        else:
            area = area_px
            perim = perim_px
            feret_max = feret_max_px
            feret_min = feret_min_px
            unit_area = "px^2"
            unit_len = "px"

        rows.append({
            "label_id": lbl,
            "instance_id": lbl,

            f"area({unit_area})": area,
            f"perimeter({unit_len})": perim,
            "circularity": circ,
            "aspect_ratio(major/minor)": aspect,
            f"feret_max({unit_len})": feret_max,
            f"feret_min({unit_len})": feret_min,
            "solidity": solidity,
            "irregularity(perim/convex_perim)": irregularity,
            "boundary_fractal_dim": fd,
            "boundary_radial_std": rad_std,
            "components_in_label": comp_count,

            "bf_mean": bf_mean,
            "bf_var": bf_var,
            "bf_entropy_bits": entropy,
            "bf_glcm_contrast": glcm["contrast"],
            "bf_glcm_dissimilarity": glcm["dissimilarity"],
            "bf_glcm_homogeneity": glcm["homogeneity"],
            "bf_glcm_ASM": glcm["ASM"],
            "bf_glcm_energy": glcm["energy"],
            "bf_glcm_correlation": glcm["correlation"],
        })

    return pd.DataFrame(rows)



# -------------------- IO & driver --------------------
def _read_gray(path: Path):
    img = skio.imread(str(path))
    if img.ndim == 3:  # RGB->gray
        img = rgb2gray(img)
    img = img.astype(np.float32)
    return img



def _read_instance_label(path: Path):
    """
    Read the instance-level label mask.
    Requirements:
    0 = background
    1, 2, 3,... = different objects

    Note: cc_label(m > 0) is not allowed here, otherwise it will be renumbered and instance_id will not match Excel.
    """
    m = skio.imread(str(path))

    if m.ndim > 2:
        m = m[..., 0]

    m = np.asarray(m)

    m = np.rint(m).astype(np.int32)

    # Negative value insurance treatment
    m[m < 0] = 0

    return m



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True, help="CSV from pairing step")
    ap.add_argument("--out_csv", required=True, help="Output CSV for features")
    ap.add_argument("--out_xlsx", default=None, help="Optional XLSX path (written at the end)")
    ap.add_argument("--pixel_size", type=float, default=None, help="µm per pixel (optional)")
    ap.add_argument("--glcm_dist", nargs="+", type=int, default=[1], help="GLCM distances, e.g., 1 2")
    ap.add_argument("--glcm_angles", nargs="+", type=float,
                    default=[0, math.pi/4, math.pi/2, 3*math.pi/4],
                    help="GLCM angles (radians)")
    ap.add_argument("--glcm_levels", type=int, default=64, help="Quantization levels")
    # 新增参数
    ap.add_argument("--resume", action="store_true",
                    help="If set, skip pairs already present in out_csv (by token).")
    ap.add_argument("--flush_every", type=int, default=1,
                    help="Write to CSV every N pairs (default 1 = 每对都写).")
    
    ap.add_argument(
        "--death_excel",
        required=True,
        help="Excel/CSV table with image_name, instance_id, live_pixels, dead_pixels, overlap_pixels, death_rate"
    )

    ap.add_argument(
        "--death_sheet",
        default=0,
        help="Sheet name or index for death_excel if it is xlsx"
    )

    args = ap.parse_args()
    
    
    """
    read live-death info
    """

    death_path = Path(args.death_excel)

    if death_path.suffix.lower() in [".xlsx", ".xls"]:
        death_df = pd.read_excel(death_path, sheet_name=args.death_sheet)
    else:
        death_df = pd.read_csv(death_path)

    death_df.columns = death_df.columns.str.strip()

    required_death_cols = {
        "image_name",
        "instance_id",
        "instance_area",
        "live_pixels",
        "dead_pixels",
        "overlap_pixels",
        "death_rate"
    }

    miss_death = required_death_cols - set(death_df.columns)
    if miss_death:
        raise SystemExit(f"Death table missing columns: {sorted(miss_death)}")

    # death_df["image_name"] = death_df["image_name"].astype(str).apply(lambda x: Path(x).name)
    # death_df["instance_id"] = death_df["instance_id"].astype(int)

    death_df["image_name"] = death_df["image_name"].astype(str).apply(lambda x: Path(x).name)
    death_df["fov_key"] = death_df["image_name"].apply(get_fov_key)
    death_df["instance_id"] = death_df["instance_id"].astype(int)

    death_df = death_df.drop_duplicates(
        subset=["fov_key", "instance_id"],
        keep="first"
    )

    pairs = pd.read_csv(args.pairs_csv)

    
    pairs.columns = pairs.columns.str.strip()

    required_cols = {"brightfield_path", "mask_path"}
    miss = required_cols - set(pairs.columns)
    if miss:
        raise SystemExit(f"Pairs CSV missing required columns: {sorted(miss)}")

    if "token" not in pairs.columns:
        pairs["token"] = [f"pair_{i:06d}" for i in range(len(pairs))]
    

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done_tokens = set()
    header_needed = True
    if out_csv.exists():
        try:
            if args.resume:
                # Read-only token column to avoid large file memory pressure
                for chunk in pd.read_csv(out_csv, usecols=["token"], chunksize=200000):
                    done_tokens.update(chunk["token"].astype(str).tolist())
            header_needed = False
        except Exception as e:
            print(f"[WARN] resume fails to read {out_csv}, rewrite list header: {e}")
            header_needed = not out_csv.exists()

    # ===== Cache write: optionally, write disk is merged every N pairs =====
    pending = []
    written_rows = 0

    for idx, r in pairs.iterrows():
        tok = str(r["token"])
        if args.resume and tok in done_tokens:
            print(f"[SKIP] pair {idx} token={tok} already exist (out_csv)")
            continue

        bf_path = Path(str(r["brightfield_path"]))
        mk_path = Path(str(r["mask_path"]))
        if not bf_path.exists() or not mk_path.exists():
            print(f"[MISS] Skip pair {idx}: file not found -> BF:{bf_path.exists()} MK:{mk_path.exists()}")
            continue

        try:
            
            bf = _read_gray(bf_path)
            lab = _read_instance_label(mk_path)

            df = extract_features(
                lab,
                brightfield=bf,
                pixel_size=args.pixel_size,
                glcm_distances=tuple(args.glcm_dist),
                glcm_angles=tuple(args.glcm_angles),
                glcm_levels=int(args.glcm_levels)
            )

            if df.empty:
                print(f"[NOTE] pair {idx} token={tok} No valid region, skip writing.")
                continue

            # merge death table
            fov_key = get_fov_key(bf_path.name)

            df["fov_key"] = fov_key
            df["image_name"] = bf_path.name
            df["instance_id"] = df["instance_id"].astype(int)

            df = df.merge(
                death_df,
                on=["fov_key", "instance_id"],
                how="left",
                validate="one_to_one",
                suffixes=("", "_death")
            )

            df["died/all"] = df["death_rate"]
            df["dead_count"] = df["dead_pixels"]
            df["all_count"] = df["overlap_pixels"]

            n_unmatched = df["death_rate"].isna().sum()
            if n_unmatched > 0:
                print(
                    f"[WARN] pair {idx} token={tok}: "
                    f"{n_unmatched}/{len(df)} instances not matched in death table. "
                    f"fov_key={fov_key}, image_name={bf_path.name}"
                )
            else:
                print(
                    f"[OK] pair {idx} token={tok}: "
                    f"all {len(df)} instances matched. fov_key={fov_key}"
                )

            # add metadata from pairs.csv
            meta_cols = [
                "drugname",
                "plate",
                "token",
                "row",
                "col",
                "folder_name",
                "brightfield_path",
                "mask_path"
            ]

            for col in meta_cols:
                if col in pairs.columns:
                    df[col] = r[col]
                else:
                    df[col] = np.nan

            pending.append(df)

            # When the flush condition is reached, the disk is dropped
            if len(pending) >= max(1, args.flush_every):
                out_block = pd.concat(pending, axis=0, ignore_index=True)
                out_block.to_csv(out_csv, mode=("a" if out_csv.exists() and not header_needed else "w"),
                                 header=header_needed, index=False, encoding="utf-8")
                written_rows += len(out_block)
                header_needed = False
                pending.clear()
                print(f"[APPEND] -> {out_csv} (+{len(out_block)} rows, total_written~{written_rows})")

        except Exception as e:
            print(f"[ERR] pair {idx} token={tok}: {e}")

    # Handling the tail
    if pending:
        out_block = pd.concat(pending, axis=0, ignore_index=True)
        out_block.to_csv(out_csv, mode=("a" if out_csv.exists() and not header_needed else "w"),
                         header=header_needed, index=False, encoding="utf-8")
        written_rows += len(out_block)
        header_needed = False
        pending.clear()
        print(f"[APPEND] -> {out_csv} (+{len(out_block)} rows, total_written~{written_rows})")

    if written_rows == 0 and not out_csv.exists():
        raise SystemExit("No features extracted.")

    print(f"[DONE] features CSV -> {out_csv}")

    # if excel
    if args.out_xlsx:
        try:
            out_xlsx = Path(args.out_xlsx)
            out_xlsx.parent.mkdir(parents=True, exist_ok=True)
            df_all = pd.read_csv(out_csv)
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                df_all.to_excel(writer, index=False, sheet_name="features")
            print(f"[SAVED] features XLSX -> {out_xlsx}")
        except Exception as e:
            print(f"[WARN] Failed to write to XLSX (CSV saved): {e}")


if __name__ == "__main__":
    main()