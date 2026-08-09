import os
import shutil
import random

# 父文件夹路径
parent_folder = "/storage/jinxinLab/wangyimin/Cell_seg/transfiner-main/datasets/val/Test"
# 新建 Filter 保存路径
filter_folder = os.path.join(parent_folder, "Filter")
os.makedirs(filter_folder, exist_ok=True)

# 遍历子文件夹
for subfolder in os.listdir(parent_folder):
    subfolder_path = os.path.join(parent_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # 获取该子文件夹的所有图片
    img_files = [f for f in os.listdir(subfolder_path)
                 if f.lower().endswith((".jpg", ".png", ".jpeg", ".tif", ".tiff")) and not f.lower().startswith('._')]

    # 随机选择不超过 20 张
    num_select = min(20, len(img_files))
    selected_imgs = random.sample(img_files, num_select)

    # 创建子文件夹保存到 Filter 中
    dest_subfolder = os.path.join(filter_folder, subfolder)
    os.makedirs(dest_subfolder, exist_ok=True)

    for img_name in selected_imgs:
        src_path = os.path.join(subfolder_path, img_name)
        dst_path = os.path.join(dest_subfolder, img_name)
        shutil.copy(src_path, dst_path)

print(f"Filter images saved to {filter_folder}")