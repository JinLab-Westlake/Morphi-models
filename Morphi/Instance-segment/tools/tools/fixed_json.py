import json
import os

def fix_coco_json(input_json, output_json):
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    new_annos = []
    removed_count = 0

    print(f"Processing {len(data['annotations'])} annotations...")

    for anno in data['annotations']:
        is_valid = True
        segm = anno.get("segmentation", [])
        
        # 1. 基础检查：必须是列表且不为空
        if not isinstance(segm, list) or len(segm) == 0:
            is_valid = False
        else:
            fixed_segm = []
            for poly in segm:
                # Check if the number of points in the polygon is >= 6
                if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0:
                    try:
                        # Force to float to filter out non-numeric content
                        poly_float = [float(x) for x in poly]
                        fixed_segm.append(poly_float)
                    except (ValueError, TypeError):
                        is_valid = False
                else:
                    is_valid = False
            
            anno["segmentation"] = fixed_segm

        if is_valid and len(anno["segmentation"]) > 0:
            new_annos.append(anno)
        else:
            removed_count += 1

    # Update JSON object
    data['annotations'] = new_annos
    
    with open(output_json, 'w') as f:
        json.dump(data, f)
    
    print(f"Done.")
    print(f"keep: {len(new_annos)} ")
    print(f"removed: {removed_count} ")
    print(f"update file saved to: {output_json}")

# main
input_path = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train_pre.json"
output_path = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/16k-DATASETS/train_fiexd.json"
fix_coco_json(input_path, output_path)