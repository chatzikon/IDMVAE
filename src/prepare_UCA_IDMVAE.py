import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

UCA_ROOT = Path(
    "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/"
    "archive/UCA Image Dataset"
)

DATA_ROOT = UCA_ROOT / "UCA_Frame_data"

SPLIT_DIRS = {
    "train": DATA_ROOT / "train",
    "val": DATA_ROOT / "validation",
    "test": DATA_ROOT / "test",
}

OUTPUT_DIR = UCA_ROOT / "processed"

IMAGE_SIZE = 256


# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

image_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            antialias=True,
        ),
        transforms.ToTensor(),
    ]
)


def load_image(image_path: Path) -> torch.Tensor:
    """
    Load every image as a three-channel float tensor in [0, 1].

    Grayscale images are converted to RGB because the IDMVAE image
    network expects tensors with shape [3, H, W]. Their original color
    type is preserved separately in labels_color.pt.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return image_transform(image)


# ---------------------------------------------------------------------
# Metadata processing
# ---------------------------------------------------------------------


def load_jsonl(metadata_path: Path) -> list[dict]:
    """Read a line-delimited JSON metadata file."""
    records: list[dict] = []

    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {metadata_path} at line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {metadata_path} at "
                    f"line {line_number}, got {type(record).__name__}."
                )

            records.append(record)

    if not records:
        raise ValueError(f"No metadata records found in: {metadata_path}")

    return records


def normalize_color_label(value: str) -> str:
    """Normalize color labels to either 'grayscale' or 'rgb'."""
    normalized = str(value).strip().lower()

    grayscale_values = {
        "grayscale",
        "greyscale",
        "grey",
        "gray",
        "visual_grayscale",
        "true_grayscale",
    }

    rgb_values = {
        "rgb",
        "colour",
        "color",
    }

    if normalized in grayscale_values:
        return "grayscale"

    if normalized in rgb_values:
        return "rgb"

    raise ValueError(f"Unknown color_type value: {value!r}")


def resolve_image_path(image_value: str, split_dir: Path) -> Path:
    """
    Resolve an image path stored in the metadata.

    The metadata normally stores paths such as:
        train/images/example.jpg
        validation/images/example.jpg
        test/images/example.jpg
    """
    raw_path = Path(str(image_value).strip())

    candidates = [
        DATA_ROOT / raw_path,
        split_dir / raw_path,
        split_dir / "images" / raw_path.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image path {image_value!r}. "
        f"Tried: {candidates}"
    )


def load_all_records() -> dict[str, list[dict]]:
    """Load and validate the official split metadata files."""
    required_fields = {
        "annotation_id",
        "image",
        "caption",
        "label",
        "color_type",
    }

    records_by_split: dict[str, list[dict]] = {}

    for split_name, split_dir in SPLIT_DIRS.items():
        if not split_dir.is_dir():
            raise NotADirectoryError(
                f"Split directory does not exist: {split_dir}"
            )

        metadata_path = split_dir / "metadata.jsonl"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata file does not exist: {metadata_path}"
            )

        records = load_jsonl(metadata_path)

        for index, record in enumerate(records):
            missing_fields = required_fields - set(record)

            if missing_fields:
                raise ValueError(
                    f"Missing fields {sorted(missing_fields)} in "
                    f"{metadata_path}, record {index}."
                )

            declared_split = str(
                record.get("split", split_name)
            ).strip().lower()

            split_aliases = {
                "train": "train",
                "training": "train",
                "val": "val",
                "validation": "val",
                "valid": "val",
                "dev": "val",
                "test": "test",
                "testing": "test",
            }

            if declared_split not in split_aliases:
                raise ValueError(
                    f"Unknown split value {declared_split!r} in "
                    f"{metadata_path}, record {index}."
                )

            normalized_split = split_aliases[declared_split]

            if normalized_split != split_name:
                raise ValueError(
                    f"Split mismatch in {metadata_path}, record {index}: "
                    f"folder is {split_name!r}, metadata says "
                    f"{declared_split!r}."
                )

        records_by_split[split_name] = records

    return records_by_split


# ---------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records_by_split = load_all_records()

    all_records = [
        record
        for split_name in ("train", "val", "test")
        for record in records_by_split[split_name]
    ]

    categories = sorted(
        {str(record["label"]).strip() for record in all_records}
    )

    category_to_id = {
        category: index
        for index, category in enumerate(categories)
    }

    id_to_category = {
        index: category
        for category, index in category_to_id.items()
    }

    color_to_id = {
        "grayscale": 0,
        "rgb": 1,
    }

    id_to_color = {
        0: "grayscale",
        1: "rgb",
    }

    # Preallocate the final tensor to avoid keeping both a Python list
    # of images and a stacked tensor in memory simultaneously.
    total_images = len(all_records)
    images_tensor = torch.empty(
        (total_images, 3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.float32,
    )
    captions: list[list[str]] = []
    category_labels: list[int] = []
    color_labels: list[int] = []
    annotation_ids: list[str] = []
    image_paths: list[str] = []
    video_ids: list[str] = []
    split_names: list[str] = []

    split_indices: dict[str, list[int]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    global_index = 0

    for split_name in ("train", "val", "test"):
        split_dir = SPLIT_DIRS[split_name]
        records = records_by_split[split_name]

        for record in tqdm(
            records,
            desc=f"Preparing UCA {split_name} data",
        ):
            image_path = resolve_image_path(
                image_value=record["image"],
                split_dir=split_dir,
            )

            caption = str(record["caption"]).strip()
            category = str(record["label"]).strip()
            color_name = normalize_color_label(record["color_type"])

            if not caption:
                raise ValueError(
                    f"Empty caption for annotation "
                    f"{record['annotation_id']!r}."
                )

            images_tensor[global_index].copy_(load_image(image_path))

            # The existing IDMVAE loader expects a list of captions per
            # image. UCA supplies one caption per image.
            captions.append([caption])

            category_labels.append(category_to_id[category])
            color_labels.append(color_to_id[color_name])
            annotation_ids.append(str(record["annotation_id"]))
            image_paths.append(str(image_path.relative_to(DATA_ROOT)))
            video_ids.append(str(record.get("video", "")))
            split_names.append(split_name)

            split_indices[split_name].append(global_index)
            global_index += 1

    # -------------------------------------------------------------
    # Construct aligned tensors
    # -------------------------------------------------------------

    labels_category = torch.tensor(
        category_labels,
        dtype=torch.long,
    )

    labels_color = torch.tensor(
        color_labels,
        dtype=torch.long,
    )

    # Compatibility alias for the CUB-specific IDMVAE code.
    labels_cluster = labels_category.clone()

    image_ids = torch.arange(
        len(images_tensor),
        dtype=torch.long,
    )

    train_indices = np.asarray(split_indices["train"], dtype=np.int64)
    validation_indices = np.asarray(split_indices["val"], dtype=np.int64)
    test_indices = np.asarray(split_indices["test"], dtype=np.int64)

    # -------------------------------------------------------------
    # Save files expected by the existing IDMVAE code
    # -------------------------------------------------------------

    torch.save(images_tensor, OUTPUT_DIR / "images.pt")
    torch.save(captions, OUTPUT_DIR / "captions.pt")
    torch.save(labels_category, OUTPUT_DIR / "labels_category.pt")
    torch.save(labels_cluster, OUTPUT_DIR / "labels_cluster.pt")
    torch.save(labels_color, OUTPUT_DIR / "labels_color.pt")
    torch.save(image_ids, OUTPUT_DIR / "image_ids.pt")

    np.save(OUTPUT_DIR / "train_idx.npy", train_indices)
    np.save(OUTPUT_DIR / "train_cluster_idx.npy", train_indices)
    np.save(OUTPUT_DIR / "val_cluster_idx.npy", validation_indices)
    np.save(OUTPUT_DIR / "test_cluster_idx.npy", test_indices)
    np.save(OUTPUT_DIR / "val_idx.npy", validation_indices)
    np.save(OUTPUT_DIR / "test_idx.npy", test_indices)

    metadata = {
        "category_to_id": category_to_id,
        "id_to_category": id_to_category,
        "num_categories": len(category_to_id),
        "color_to_id": color_to_id,
        "id_to_color": id_to_color,
        "annotation_ids": annotation_ids,
        "image_paths": image_paths,
        "video_ids": video_ids,
        "splits": split_names,
    }

    with (OUTPUT_DIR / "metadata.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print("\nDataset successfully prepared")
    print("-----------------------------")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Images:           {len(images_tensor)}")
    print(f"Image shape:      {tuple(images_tensor.shape)}")
    print(
        f"Tensor size: "
        f"{images_tensor.numel()*images_tensor.element_size()/(1024**3):.2f} GiB"
    )
    print(f"Caption pairs:    {len(captions)}")
    print(f"Captions/image:   {{1: {len(captions)}}}")
    print(f"Crime classes:    {category_to_id}")
    print(f"Number of classes:{len(category_to_id):>5}")
    print(f"Color classes:    {color_to_id}")

    print("\nSplit sizes")
    print("-----------")
    print(f"Train:      {len(train_indices)} images")
    print(f"Validation: {len(validation_indices)} images")
    print(f"Test:       {len(test_indices)} images")

    category_counts = Counter(
        str(record["label"]).strip()
        for record in all_records
    )

    color_counts = Counter(
        normalize_color_label(record["color_type"])
        for record in all_records
    )

    print("\nCategory distribution")
    print("---------------------")
    for category in categories:
        print(f"{category:<20} {category_counts[category]}")

    print("\nColor distribution")
    print("------------------")
    for color_name in ("rgb", "grayscale"):
        print(f"{color_name:<20} {color_counts[color_name]}")


if __name__ == "__main__":
    main()