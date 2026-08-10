import argparse
import os

import numpy as np
from PIL import Image


def convert_semantic_mask(
    input_path,
    output_path
):
    """
    Convert DoANet raw prediction labels:

        0 -> 0      background
        1 -> 127    live
        2 -> 254    dead

    Parameters
    ----------
    input_path : str
        Path to DoANet raw prediction mask.

    output_path : str
        Path to save converted semantic mask.
    """

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input mask not found: {input_path}"
        )

    # --------------------------------------------------------
    # Read mask
    # --------------------------------------------------------

    mask = np.array(
        Image.open(input_path)
    )

    # If unexpectedly RGB/RGBA, use first channel
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # --------------------------------------------------------
    # Validate labels
    # --------------------------------------------------------

    unique_labels = np.unique(mask)

    print(
        f"Input mask: {input_path}"
    )

    print(
        f"Input shape: {mask.shape}"
    )

    print(
        f"Raw DoANet labels: {unique_labels}"
    )

    allowed_labels = {
        0,
        1,
        2
    }

    actual_labels = set(
        unique_labels.tolist()
    )

    if not actual_labels.issubset(
        allowed_labels
    ):
        raise ValueError(
            "Unexpected labels in DoANet prediction.\n"
            f"Found: {unique_labels}\n"
            "Expected only: 0, 1, 2"
        )

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    converted = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    converted[mask == 0] = 0
    converted[mask == 1] = 127
    converted[mask == 2] = 254

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        os.path.abspath(output_path)
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    Image.fromarray(
        converted
    ).save(
        output_path
    )

    # --------------------------------------------------------
    # Verify saved labels
    # --------------------------------------------------------

    converted_labels = np.unique(
        converted
    )

    print(
        f"Converted labels: {converted_labels}"
    )

    print(
        f"Saved semantic mask: {output_path}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert DoANet raw semantic prediction "
            "from 0/1/2 to 0/127/254."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="DoANet raw prediction mask"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Converted output mask"
    )

    args = parser.parse_args()

    convert_semantic_mask(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()