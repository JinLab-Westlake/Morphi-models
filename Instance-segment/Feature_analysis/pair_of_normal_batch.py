#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair Brightfield & Mask images by rXXcYY token and export a mapping table.

Usage:
  python pair_bf_mask.py \
    --bf_dir  /storage/.../Combined/channel_3 \
    --mask_dir /storage/.../Combined/channel_3/output/mask \
    --out_csv  /path/to/pairs.csv \
    [--out_xlsx /path/to/pairs.xlsx]
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


TOKEN_PAT = re.compile(
    r"""r(?P<r>\d{1,2})c(?P<c>\d{1,2})f(?P<f>\d{1,3})""",
    re.IGNORECASE,
)


def normalize_f(f_raw: str) -> str:
    """
    Normalize the view/fov numbers.
    f1  -> 01
    f01 -> 01
    f001 -> 01
    fA1 -> A01
    """
    f_raw = f_raw.upper()

    m = re.fullmatch(r"([A-Z]?)(\d{1,3})", f_raw)
    if not m:
        return f_raw

    prefix, num = m.groups()
    return f"{prefix}{int(num):02d}"


def parse_token_from_filename(filename: str) -> Tuple[str, int, int, str]:
    """
    Parse r/c/f from the filename and return:
    token, row, col, view

    token format:
    rXXcYYfXX
    """
    m = TOKEN_PAT.search(filename)
    if not m:
        return None, None, None, None

    r = int(m.group("r"))
    c = int(m.group("c"))
    f = normalize_f(m.group("f"))

    token = f"r{r:02d}c{c:02d}f{f}"
    return token, r, c, f


def find_paths_by_token(folder: Path) -> Dict[tuple, Path]:
    """
    Make a dictionary of (token, view) -> full path to the file.
    """
    mapping: Dict[tuple, Path] = {}
    dup_warn: Dict[tuple, List[Path]] = {}

    for root, _, files in os.walk(folder):
        for fn in files:
            ext = Path(fn).suffix.lower()
            if ext not in IMG_EXTS:
                continue

            m = TOKEN_PAT.search(fn)
            if not m:
                continue

            r = int(m.group("r"))
            c = int(m.group("c"))
            f = m.group("f")

            if f is None:
                continue

            token = f"r{r:02d}c{c:02d}"
            view = f"{int(f):02d}"

            key = (token, view)

            p = Path(root) / fn

            if key in mapping:
                prev = mapping[key]
                if str(p) < str(prev):
                    mapping[key] = p
                dup_warn.setdefault(key, [prev]).append(p)
            else:
                mapping[key] = p

    if dup_warn:
        print(
            f"[WARN] {folder}: {len(dup_warn)} token-view pairs have multiple files; "
            "took lexicographically first."
        )

    return mapping


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--bf_dir",
        required=True,
        help="Brightfield image root directory, scanned recursively",
    )
    ap.add_argument(
        "--mask_dir",
        required=True,
        help="Mask image root directory, scanned recursively",
    )
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument(
        "--out_xlsx",
        default=None,
        help="Optional XLSX path, requires pandas and openpyxl",
    )

    args = ap.parse_args()

    bf_dir = Path(args.bf_dir)
    mask_dir = Path(args.mask_dir)

    if not bf_dir.exists():
        raise SystemExit(f"Brightfield directory not found: {bf_dir}")
    if not mask_dir.exists():
        raise SystemExit(f"Mask directory not found: {mask_dir}")

    print(f"[INFO] Scanning Brightfield in: {bf_dir}")
    bf_map = find_paths_by_token(bf_dir)

    print(f"[INFO] Scanning Mask in: {mask_dir}")
    mk_map = find_paths_by_token(mask_dir)

    if not bf_map and not mk_map:
        raise SystemExit("No valid rXXcYYfXX images found in both directories.")

    rows = []
    n_pairs = 0
    n_missed_bf = 0
    n_missed_mask = 0

    tokens = sorted(set(bf_map.keys()) | set(mk_map.keys()))

    for key in tokens:
        token, view = key

        bf_p = bf_map.get(key)
        mk_p = mk_map.get(key)

        if bf_p is None:
            n_missed_bf += 1
            continue

        if mk_p is None:
            n_missed_mask += 1
            continue

        row = int(token[1:3])
        col = int(token[4:6])

        well_id = token
        fov_id = f"{token}f{view}"

        rows.append(
            {
                "token": token,
                "row": row,
                "col": col,
                "view": view,
                "well_id": well_id,
                "fov_id": fov_id,
                "brightfield_path": str(bf_p),
                "mask_path": str(mk_p),
            }
        )

        n_pairs += 1

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "token",
        "row",
        "col",
        "view",
        "well_id",
        "fov_id",
        "brightfield_path",
        "mask_path",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
            
    print(
        f"[DONE] matched_pairs={n_pairs}, "
        f"missing_brightfield={n_missed_bf}, missing_mask={n_missed_mask}"
    )
    print(f"[SAVED] CSV -> {out_csv}")

    if args.out_xlsx:
        if pd is None:
            raise SystemExit(
                "Writing XLSX requires pandas and openpyxl: pip install pandas openpyxl"
            )

        df = pd.DataFrame(rows, columns=fieldnames)

        xlsx_path = Path(args.out_xlsx)
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="pairs")

        print(f"[SAVED] XLSX -> {xlsx_path}")


if __name__ == "__main__":
    main()