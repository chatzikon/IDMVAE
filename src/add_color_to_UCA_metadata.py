#!/usr/bin/env python3

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# UCA_OUTPUT_ROOT = Path(
#     "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA Image Dataset/UCA_Frame_data"
# )

UCA_OUTPUT_ROOT = Path(
    "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset"
)

SPLIT_METADATA_FILES = {
    "train": UCA_OUTPUT_ROOT / "train" / "metadata.jsonl",
    "validation": UCA_OUTPUT_ROOT / "validation" / "metadata.jsonl",
    "test": UCA_OUTPUT_ROOT / "test" / "metadata.jsonl",
}


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Keep a copy of each original metadata file before replacing it.
CREATE_BACKUP = True

# Thresholds for detecting images that are stored as RGB but visually
# appear grayscale.
MEAN_CHANNEL_DIFFERENCE_THRESHOLD = 3.0
P95_CHANNEL_DIFFERENCE_THRESHOLD = 8.0


# ---------------------------------------------------------------------
# Color classification
# ---------------------------------------------------------------------

def classify_color(image_path: Path) -> str:
    """
    Classify an image using exactly two labels:

        grayscale
        rgb

    An image is considered grayscale when:

    1. It is physically stored as a single-channel grayscale image, or
    2. It is stored as RGB, but its color channels are sufficiently similar.
    """

    with Image.open(image_path) as image:
        original_mode = image.mode

        # Images physically stored with one grayscale channel.
        if original_mode in {"1", "L", "I", "I;16", "F"}:
            return "grayscale"

        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.int16,
        )

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # Largest color-channel difference at each pixel.
    pixel_channel_difference = np.maximum.reduce(
        [
            np.abs(red - green),
            np.abs(red - blue),
            np.abs(green - blue),
        ]
    )

    mean_difference = float(
        pixel_channel_difference.mean()
    )

    p95_difference = float(
        np.percentile(
            pixel_channel_difference,
            95,
        )
    )

    if (
        mean_difference
        <= MEAN_CHANNEL_DIFFERENCE_THRESHOLD
        and p95_difference
        <= P95_CHANNEL_DIFFERENCE_THRESHOLD
    ):
        return "grayscale"

    return "rgb"


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------

def resolve_image_path(record: dict) -> Path:
    """
    Resolve the image path stored in one metadata record.

    The extraction script stores image paths relative to UCA_OUTPUT_ROOT,
    for example:

        train/images/Abuse001_0000_ab12cd34ef56.jpg
    """

    image_value = record.get("image")

    if image_value is None:
        raise KeyError(
            "Metadata record does not contain an 'image' field."
        )

    image_value = str(image_value).strip()

    if not image_value:
        raise ValueError(
            "Metadata record contains an empty 'image' field."
        )

    image_path = Path(image_value)

    if not image_path.is_absolute():
        image_path = UCA_OUTPUT_ROOT / image_path

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file does not exist: {image_path}"
        )

    return image_path


def backup_path_for(metadata_path: Path) -> Path:
    """
    Example:

        metadata.jsonl
        metadata.jsonl.backup
    """

    return metadata_path.with_name(
        metadata_path.name + ".backup"
    )


def temporary_path_for(metadata_path: Path) -> Path:
    """
    Example:

        metadata.jsonl
        metadata.jsonl.tmp
    """

    return metadata_path.with_name(
        metadata_path.name + ".tmp"
    )


# ---------------------------------------------------------------------
# Split processing
# ---------------------------------------------------------------------

def process_metadata_file(
    split_name: str,
    metadata_path: Path,
) -> Counter:
    """
    Add or replace 'color_type' in every JSONL record.

    The original file is replaced only after the complete split has been
    processed successfully.
    """

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{split_name} metadata file not found: "
            f"{metadata_path}"
        )

    temporary_path = temporary_path_for(metadata_path)
    backup_path = backup_path_for(metadata_path)

    # Remove an old temporary file from a previous interrupted execution.
    if temporary_path.exists():
        temporary_path.unlink()

    counts = Counter()
    processed_records = 0

    try:
        with (
            metadata_path.open(
                "r",
                encoding="utf-8",
            ) as input_file,
            temporary_path.open(
                "w",
                encoding="utf-8",
            ) as output_file,
        ):
            for line_number, line in enumerate(
                input_file,
                start=1,
            ):
                line = line.strip()

                # Preserve the convention of ignoring empty JSONL lines.
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {metadata_path}, "
                        f"line {line_number}: {error}"
                    ) from error

                if not isinstance(record, dict):
                    raise ValueError(
                        f"{metadata_path}, line {line_number}: "
                        "expected a JSON object."
                    )

                image_path = resolve_image_path(record)

                try:
                    color_type = classify_color(image_path)
                except (OSError, ValueError) as error:
                    raise RuntimeError(
                        f"Could not classify image at "
                        f"{metadata_path}, line {line_number}: "
                        f"{image_path}. Error: {error}"
                    ) from error

                # Add the field, or replace it if it already exists.
                record["color_type"] = color_type

                counts[color_type] += 1
                processed_records += 1

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Create a backup only after the new temporary file has been
        # written successfully.
        if CREATE_BACKUP:
            shutil.copy2(
                metadata_path,
                backup_path,
            )

        # Safely replace the original metadata file.
        temporary_path.replace(metadata_path)

    except Exception:
        # Do not leave a partially written temporary file behind.
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    print()
    print(split_name)
    print("-" * len(split_name))
    print(f"Metadata file: {metadata_path}")
    print(f"Processed:     {processed_records}")
    print(f"RGB:           {counts['rgb']}")
    print(f"Grayscale:     {counts['grayscale']}")

    if CREATE_BACKUP:
        print(f"Backup:        {backup_path}")

    return counts


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if not UCA_OUTPUT_ROOT.is_dir():
        raise NotADirectoryError(
            f"UCA output directory does not exist: "
            f"{UCA_OUTPUT_ROOT}"
        )

    total_counts = Counter()
    total_records = 0

    for split_name, metadata_path in SPLIT_METADATA_FILES.items():
        split_counts = process_metadata_file(
            split_name=split_name,
            metadata_path=metadata_path,
        )

        total_counts.update(split_counts)
        total_records += sum(split_counts.values())

    print()
    print("Overall")
    print("-------")
    print(f"Processed: {total_records}")
    print(f"RGB:       {total_counts['rgb']}")
    print(f"Grayscale: {total_counts['grayscale']}")


if __name__ == "__main__":
    main()