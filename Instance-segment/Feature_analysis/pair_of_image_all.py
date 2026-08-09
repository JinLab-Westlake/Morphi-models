import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import pandas as pd
except Exception:
    pd = None  # only xlsx need pandas/openpyxl

# Compatible with multiple image suffixes
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Match the experiment folder and extract drug and plate
FOLDER_PAT = re.compile(
    r"""^(?P<date>\d{8})_(?P<drug>.+?)_NO(?P<plate>\d+)_P\d{2}-P\d{2}__.+?-Measurement_1$""",
    re.IGNORECASE
)

# Match rXXcYY in the filename (f suffix is optional, number of bits can be 1-2)
TOKEN_PAT = re.compile(
    r"""r(?P<r>\d{1,2})c(?P<c>\d{1,2})(?:f(?P<f>\d{1,3}|[A-Za-z]\d{1,3}))?""",
    re.IGNORECASE
)

def find_paths_by_token(folder: Path) -> Dict[str, Path]:
    """
    Recursively search in the given experiment folder to build a dictionary of the full path of the token -> file.
    The token is normalized with zero padding: rXXcYY (ignoring the f part).
    If multiple files corresponding to the same token are found, the path with the smallest alphabetical order will be preferred and an alarm will be printed.
    """
    mapping: Dict[str, Path] = {}
    dup_warn: Dict[str, List[Path]] = {}
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
            token = f"r{r:02d}c{c:02d}"
            p = Path(root) / fn
            if token in mapping:
                prev = mapping[token]
                if str(p) < str(prev):
                    mapping[token] = p
                dup_warn.setdefault(token, [prev]).append(p)
            else:
                mapping[token] = p
    if dup_warn:
        print(f"[WARN] {folder.name}: {len(dup_warn)} tokens have multiple files; took lexicographically first.")
    return mapping

def list_measurement_folders(base: Path) -> Dict[str, Tuple[str, int, Path]]:
    """
    Return all experiments in base that conform to the naming convention:
      key = folder_name,
      value = (drugname, plate, folder_path)
    """
    out: Dict[str, Tuple[str, int, Path]] = {}
    if not base.exists():
        return out
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = FOLDER_PAT.match(p.name)
        if not m:
            continue
        drug = m.group("drug")
        plate = int(m.group("plate"))
        out[p.name] = (drug, plate, p)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Parent folder containing Brightfield/Brightfiled and Mask subfolders")
    ap.add_argument("--out_csv",  required=True, help="Output CSV path")
    ap.add_argument("--out_xlsx", default=None, help="Optional XLSX path (requires pandas/openpyxl)")
    args = ap.parse_args()

    root = Path(args.root_dir)
    
    print(root)

    data_folders = list_measurement_folders(root)

    rows = []
    n_pairs = 0
    n_missed_bf = 0
    n_missed_mask = 0
    
    

    for fname in data_folders:
        
        celline_og, plate_og, og_path = data_folders[fname]
        # print(og_path)
        # print(celline_og)
        
        print(og_path)
        
        bf_path = og_path / "channel/flat_field/pct"
        mk_path = og_path / "channel/output/instance_mask"
        
        print(bf_path)
        
        drug = celline_og
        plate = plate_og

        # Creating a token map
        bf_map = find_paths_by_token(bf_path)
        mk_map = find_paths_by_token(mk_path)
        
        # print(bf_map)

        tokens = sorted(set(bf_map.keys()) | set(mk_map.keys()))
        for tok in tokens:
            bf_p = bf_map.get(tok)
            mk_p = mk_map.get(tok)
            if bf_p is None:
                n_missed_bf += 1
                continue
            if mk_p is None:
                n_missed_mask += 1
                continue

            # Parsing rows and columns
            m = TOKEN_PAT.search(tok)
            if m:
                r = int(m.group("r")) if m.group("r") else int(tok[1:3])
                c = int(m.group("c")) if m.group("c") else int(tok[4:6])
            else:
                # tok is already in the form rXXcYY
                try:
                    r = int(tok[1:3]); c = int(tok[4:6])
                except Exception:
                    r = c = -1

            rows.append({
                "drugname": drug,
                "plate": f"NO{plate}",
                "token": tok,
                "row": r,
                "col": c,
                "folder_name": fname,
                "brightfield_path": str(bf_p),
                "mask_path": str(mk_p),
            })
            n_pairs += 1

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "drugname","plate","token","row","col","folder_name","brightfield_path","mask_path"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[DONE] matched_pairs={n_pairs}, missing_brightfield={n_missed_bf}, missing_mask={n_missed_mask}")
    print(f"[SAVED] CSV -> {out_csv}")

    # choose XLSX
    if args.out_xlsx:
        if pd is None:
            raise SystemExit("Writing XLSX requires pandas: pip install pandas openpyxl")
        df = pd.DataFrame(rows, columns=[
            "drugname","plate","token","row","col","folder_name","brightfield_path","mask_path"
        ])
        xlsx_path = Path(args.out_xlsx)
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="pairs")
        print(f"[SAVED] XLSX -> {xlsx_path}")

if __name__ == "__main__":
    main()
