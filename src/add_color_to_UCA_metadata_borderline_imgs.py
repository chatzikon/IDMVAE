#!/usr/bin/env python3

import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


UCA_OUTPUT_ROOT = Path(
    "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/"
    "archive/UCA Image Dataset/UCA_Frame_data"
)

SPLITS = ["train", "validation", "test"]

MEAN_THRESHOLD = 3.0
P95_THRESHOLD = 8.0



# Number of closest images to review.
NUM_BORDERLINE_IMAGES = 100

REVIEW_ROOT = UCA_OUTPUT_ROOT / "borderline_color_review"


def calculate_color_scores(image_path: Path) -> tuple[float, float]:
    with Image.open(image_path) as image:
        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.int16,
        )

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    channel_difference = np.maximum.reduce(
        [
            np.abs(red - green),
            np.abs(red - blue),
            np.abs(green - blue),
        ]
    )

    mean_difference = float(channel_difference.mean())
    p95_difference = float(
        np.percentile(channel_difference, 95)
    )

    return mean_difference, p95_difference


def classify_color(
    mean_difference: float,
    p95_difference: float,
) -> str:
    if (
        mean_difference <= MEAN_THRESHOLD
        and p95_difference <= P95_THRESHOLD
    ):
        return "grayscale"

    return "rgb"


def boundary_distance(
    mean_difference: float,
    p95_difference: float,
) -> float:
    """
    Returns a normalized distance from the decision boundary.

    Values close to zero are the most borderline.
    """

    mean_margin = (
        mean_difference - MEAN_THRESHOLD
    ) / MEAN_THRESHOLD

    p95_margin = (
        p95_difference - P95_THRESHOLD
    ) / P95_THRESHOLD

    # The image is classified as grayscale only when both margins
    # are <= 0. The largest margin determines how close it is to
    # crossing the AND-based decision boundary.
    return abs(max(mean_margin, p95_margin))


def resolve_image_path(record: dict) -> Path:
    image_path = Path(record["image"])

    if not image_path.is_absolute():
        image_path = UCA_OUTPUT_ROOT / image_path

    return image_path


def create_preview(
    image_path: Path,
    output_path: Path,
    split: str,
    label: str,
    mean_difference: float,
    p95_difference: float,
) -> None:
    with Image.open(image_path) as image:
        preview = image.convert("RGB")
        preview.thumbnail((800, 600))

    text_height = 70

    canvas = Image.new(
        "RGB",
        (
            preview.width,
            preview.height + text_height,
        ),
        "white",
    )

    canvas.paste(preview, (0, 0))

    draw = ImageDraw.Draw(canvas)

    description = (
        f"split={split} | predicted={label}\n"
        f"mean_difference={mean_difference:.3f} "
        f"(threshold={MEAN_THRESHOLD}) | "
        f"p95_difference={p95_difference:.3f} "
        f"(threshold={P95_THRESHOLD})"
    )

    draw.text(
        (10, preview.height + 10),
        description,
        fill="black",
    )

    canvas.save(output_path, quality=95)


def main() -> None:
    REVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for split in SPLITS:
        metadata_path = (
            UCA_OUTPUT_ROOT
            / split
            / "metadata.jsonl"
        )

        if not metadata_path.is_file():
            print(f"Skipping missing file: {metadata_path}")
            continue

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)
                image_path = resolve_image_path(record)

                if not image_path.is_file():
                    print(f"Missing image: {image_path}")
                    continue

                mean_difference, p95_difference = (
                    calculate_color_scores(image_path)
                )

                label = classify_color(
                    mean_difference,
                    p95_difference,
                )

                distance = boundary_distance(
                    mean_difference,
                    p95_difference,
                )

                results.append(
                    {
                        "split": split,
                        "line_number": line_number,
                        "image_path": image_path,
                        "image": record.get("image", ""),
                        "predicted_label": label,
                        "mean_difference": mean_difference,
                        "p95_difference": p95_difference,
                        "boundary_distance": distance,
                    }
                )

    results.sort(
        key=lambda item: item["boundary_distance"]
    )

    selected = results[:NUM_BORDERLINE_IMAGES]

    csv_path = REVIEW_ROOT / "borderline_images.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "rank",
                "split",
                "line_number",
                "image",
                "predicted_label",
                "mean_difference",
                "p95_difference",
                "boundary_distance",
                "review_label",
            ],
        )

        writer.writeheader()

        for rank, item in enumerate(selected, start=1):
            image_path = item["image_path"]

            output_name = (
                f"{rank:03d}_"
                f"{item['predicted_label']}_"
                f"{item['split']}_"
                f"{image_path.name}"
            )

            output_path = REVIEW_ROOT / output_name

            create_preview(
                image_path=image_path,
                output_path=output_path,
                split=item["split"],
                label=item["predicted_label"],
                mean_difference=item["mean_difference"],
                p95_difference=item["p95_difference"],
            )

            writer.writerow(
                {
                    "rank": rank,
                    "split": item["split"],
                    "line_number": item["line_number"],
                    "image": item["image"],
                    "predicted_label": item["predicted_label"],
                    "mean_difference": (
                        f"{item['mean_difference']:.6f}"
                    ),
                    "p95_difference": (
                        f"{item['p95_difference']:.6f}"
                    ),
                    "boundary_distance": (
                        f"{item['boundary_distance']:.6f}"
                    ),
                    "review_label": "",
                }
            )

    print(f"Reviewed candidates: {len(selected)}")
    print(f"Preview folder: {REVIEW_ROOT}")
    print(f"CSV file:       {csv_path}")


if __name__ == "__main__":
    main()