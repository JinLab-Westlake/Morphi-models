import argparse
import os

import numpy as np
from PIL import Image


def convert_semantic_mask(input_path, output_path):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input mask not found: {input_path}")

    mask = np.array(Image.open(input_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    labels = np.unique(mask)
    # print(f"Raw labels: {labels}")

    if not set(labels).issubset({0, 1, 2}):
        raise ValueError(f"Unexpected labels: {labels}")

    converted = np.zeros_like(mask, dtype=np.uint8)
    converted[mask == 1] = 127
    converted[mask == 2] = 254

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(converted).save(output_path)

    print(f"Converted labels: {np.unique(converted)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert_semantic_mask(args.input, args.output)