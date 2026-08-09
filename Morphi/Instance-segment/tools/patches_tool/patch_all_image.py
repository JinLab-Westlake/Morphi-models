import cv2
import os
import sys


def generate_patch_coords(image_shape, patch_size=224, stride=224):
    """
    The patch coordinates of the whole graph are generated to ensure that the whole graph is covered.
    Return: [(x1, y1, x2, y2),...]
    """
    H, W = image_shape[:2]
    coords = []

    xs = list(range(0, max(W - patch_size + 1, 1), stride))
    ys = list(range(0, max(H - patch_size + 1, 1), stride))

    if len(xs) == 0 or xs[-1] != W - patch_size:
        xs.append(max(W - patch_size, 0))
    if len(ys) == 0 or ys[-1] != H - patch_size:
        ys.append(max(H - patch_size, 0))

    xs = sorted(set(xs))
    ys = sorted(set(ys))

    for y1 in ys:
        for x1 in xs:
            x2 = x1 + patch_size
            y2 = y1 + patch_size
            coords.append((x1, y1, x2, y2))

    return coords


def process_one_combined_folder(image_folder, patch_size=224, stride=224):
    output_bright = os.path.join(image_folder, f"{patch_size}_patches_all", "brightfield")
    os.makedirs(output_bright, exist_ok=True)

    coord_file = os.path.join(output_bright, "patch_coords.txt")

    print(f"\nimage folder is: {image_folder}")
    print(f"output folder is: {output_bright}")
    print(f"patch size: {patch_size}, stride: {stride}")

    with open(coord_file, "w") as f:
        f.write("filename,x1,y1,x2,y2\n")

    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(valid_exts):
            continue

        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Cannot read image {image_path}")
            continue

        print(f"Processing {filename}")

        patch_coords = generate_patch_coords(
            image_shape=image.shape,
            patch_size=patch_size,
            stride=stride
        )

        prefix = os.path.splitext(filename)[0]

        for i, (x1, y1, x2, y2) in enumerate(patch_coords):
            patch = image[y1:y2, x1:x2]

            patch_name = f"{prefix}_patch_{i}.png"
            patch_path = os.path.join(output_bright, patch_name)

            cv2.imwrite(patch_path, patch)

            with open(coord_file, "a") as f:
                f.write(f"{patch_name},{x1},{y1},{x2},{y2}\n")

        print(f"Saved {len(patch_coords)} patches for {filename}")

    print(f"Done: {image_folder}")


def main():
    default_root_folder = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20241230"

    root_folder = sys.argv[1] if len(sys.argv) > 1 else default_root_folder
    patch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 224
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 224

    print(f"Root folder: {root_folder}")
    
    # key_word = "KYSE150"
    
    # print(key_word)

    if not os.path.isdir(root_folder):
        print(f"Error: root folder does not exist: {root_folder}")
        return

    for name in sorted(os.listdir(root_folder)):
        dataset_path = os.path.join(root_folder, name)
        if not os.path.isdir(dataset_path):
            continue

        image_folder = os.path.join(dataset_path, "channel", "BF")
        output_bright = os.path.join(image_folder, f"{patch_size}_patches_all", "brightfield")

        if not os.path.isdir(image_folder):
            print(f"Skip: {image_folder} does not exist")
            continue

        # Only handle brightfield output paths that do not exist
        if os.path.exists(output_bright):
            print(f"Skip: {output_bright} already exists")
            continue

        process_one_combined_folder(
            image_folder=image_folder,
            patch_size=patch_size,
            stride=stride
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()