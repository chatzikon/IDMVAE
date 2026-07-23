import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

UCF_ROOT = Path(
    "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/"
    "archive/UCF Image Dataset"
)

IMAGE_ROOT = UCF_ROOT / "UCF Image Dataset"

ANNOTATIONS_CSV = (
    UCF_ROOT / "image_category_captions_with_color.csv"
)

OUTPUT_DIR = UCF_ROOT / "processed"

IMAGE_SIZE = 256
RANDOM_STATE = 42


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


def build_image_index(
    image_root: Path,
) -> dict[str, list[Path]]:
    """
    Create:
        filename -> list of matching paths

    A list is retained because the same filename could theoretically
    appear in more than one category directory.
    """
    image_index: dict[str, list[Path]] = {}

    for image_path in image_root.rglob("*.png"):
        image_index.setdefault(
            image_path.name,
            [],
        ).append(image_path)

    return image_index


def normalize_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def resolve_image_path(
    image_name: str,
    category: str,
    image_index: dict[str, list[Path]],
) -> Path:
    """
    Resolve an image using its filename.

    When duplicate filenames exist, prefer a path whose parent
    directory matches the category.
    """
    candidates = image_index.get(image_name, [])

    if not candidates:
        raise FileNotFoundError(
            f"Image not found: {image_name}"
        )

    if len(candidates) == 1:
        return candidates[0]

    normalized_category = normalize_name(category)

    category_matches = [
        path
        for path in candidates
        if normalize_name(path.parent.name)
        == normalized_category
    ]

    if len(category_matches) == 1:
        return category_matches[0]

    raise RuntimeError(
        f"Could not uniquely resolve {image_name}. "
        f"Candidates: {candidates}"
    )


def load_image(image_path: Path) -> torch.Tensor:
    """
    Load every input as a three-channel tensor.

    Even grayscale inputs become three-channel tensors because the
    IDMVAE image network expects [3, H, W]. Their original colour
    status remains available through labels_color.pt.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        return image_transform(image)


# ---------------------------------------------------------------------
# Caption and label processing
# ---------------------------------------------------------------------

def split_captions(value: str) -> list[str]:
    captions = [
        caption.strip()
        for caption in str(value).split("|")
        if caption.strip()
    ]

    if not captions:
        raise ValueError(
            f"No valid captions found in: {value!r}"
        )

    return captions


def normalize_color_label(value: str) -> str:
    """
    Produce only two labels:
        grayscale
        rgb
    """
    normalized = str(value).strip().lower()

    grayscale_values = {
        "grayscale",
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

    raise ValueError(
        f"Unknown color_type value: {value!r}"
    )


# ---------------------------------------------------------------------
# Stratified image-level splits
# ---------------------------------------------------------------------

def create_splits(
    labels_category: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 70/15/15 train/validation/test splits.

    Splitting is performed at image level. All captions belonging to
    the same image therefore remain in the same split.
    """
    all_indices = np.arange(len(labels_category))
    labels_numpy = labels_category.numpy()

    train_indices, temporary_indices = train_test_split(
        all_indices,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=labels_numpy,
    )

    temporary_labels = labels_numpy[temporary_indices]

    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.67,
        random_state=RANDOM_STATE,
        stratify=temporary_labels,
    )

    return (
        np.sort(train_indices),
        np.sort(validation_indices),
        np.sort(test_indices),
    )


# ---------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------

def main() -> None:
    if not IMAGE_ROOT.is_dir():
        raise NotADirectoryError(
            f"Image directory does not exist: {IMAGE_ROOT}"
        )

    if not ANNOTATIONS_CSV.is_file():
        raise FileNotFoundError(
            f"CSV does not exist: {ANNOTATIONS_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotations = pd.read_csv(ANNOTATIONS_CSV)

    required_columns = {
        "image_name",
        "category",
        "ref_captions",
        "color_type",
    }

    missing_columns = (
        required_columns - set(annotations.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing CSV columns: "
            f"{sorted(missing_columns)}"
        )

    # Preserve the CSV ordering across every output file.
    annotations = annotations.reset_index(drop=True)

    categories = sorted(
        annotations["category"]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
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

    image_index = build_image_index(IMAGE_ROOT)

    images: list[torch.Tensor] = []
    captions: list[list[str]] = []
    category_labels: list[int] = []
    color_labels: list[int] = []
    image_names: list[str] = []
    image_paths: list[str] = []

    for row_index, row in tqdm(
        annotations.iterrows(),
        total=len(annotations),
        desc="Preparing UCF data",
    ):
        image_name = str(row["image_name"]).strip()
        category = str(row["category"]).strip()

        image_path = resolve_image_path(
            image_name=image_name,
            category=category,
            image_index=image_index,
        )

        row_captions = split_captions(
            row["ref_captions"]
        )

        color_name = normalize_color_label(
            row["color_type"]
        )

        images.append(load_image(image_path))
        captions.append(row_captions)

        category_labels.append(
            category_to_id[category]
        )

        color_labels.append(
            color_to_id[color_name]
        )

        image_names.append(image_name)

        image_paths.append(
            str(image_path.relative_to(IMAGE_ROOT))
        )

    # -------------------------------------------------------------
    # Construct aligned tensors
    # -------------------------------------------------------------

    images_tensor = torch.stack(images).float()

    labels_category = torch.tensor(
        category_labels,
        dtype=torch.long,
    )

    labels_color = torch.tensor(
        color_labels,
        dtype=torch.long,
    )

    # Compatibility alias for the CUB-specific code.
    # There is no separate cluster hierarchy in UCF.
    labels_cluster = labels_category.clone()

    # Numeric IDs are safer with PyTorch's default DataLoader collate.
    image_ids = torch.arange(
        len(images_tensor),
        dtype=torch.long,
    )

    # -------------------------------------------------------------
    # Create image-level splits
    # -------------------------------------------------------------

    (
        train_indices,
        validation_indices,
        test_indices,
    ) = create_splits(labels_category)

    # -------------------------------------------------------------
    # Save files expected by the IDMVAE code
    # -------------------------------------------------------------

    torch.save(
        images_tensor,
        OUTPUT_DIR / "images.pt",
    )

    torch.save(
        captions,
        OUTPUT_DIR / "captions.pt",
    )

    torch.save(
        labels_category,
        OUTPUT_DIR / "labels_category.pt",
    )

    torch.save(
        labels_cluster,
        OUTPUT_DIR / "labels_cluster.pt",
    )

    torch.save(
        labels_color,
        OUTPUT_DIR / "labels_color.pt",
    )

    torch.save(
        image_ids,
        OUTPUT_DIR / "image_ids.pt",
    )

    # The existing CUB loader expects these particular names.
    np.save(
        OUTPUT_DIR / "train_idx.npy",
        train_indices,
    )

    np.save(
        OUTPUT_DIR / "train_cluster_idx.npy",
        train_indices,
    )

    np.save(
        OUTPUT_DIR / "val_cluster_idx.npy",
        validation_indices,
    )

    np.save(
        OUTPUT_DIR / "test_cluster_idx.npy",
        test_indices,
    )

    # Additional cleanly named aliases.
    np.save(
        OUTPUT_DIR / "val_idx.npy",
        validation_indices,
    )

    np.save(
        OUTPUT_DIR / "test_idx.npy",
        test_indices,
    )

    metadata = {
        "category_to_id": category_to_id,
        "id_to_category": id_to_category,
        "color_to_id": color_to_id,
        "id_to_color": id_to_color,
        "image_names": image_names,
        "image_paths": image_paths,
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    caption_counts = [
        len(image_captions)
        for image_captions in captions
    ]

    print("\nDataset successfully prepared")
    print("-----------------------------")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Images:           {len(images_tensor)}")
    print(f"Image shape:      {tuple(images_tensor.shape)}")
    print(f"Caption pairs:    {sum(caption_counts)}")
    print(
        "Captions/image:  "
        f"{dict(sorted(Counter(caption_counts).items()))}"
    )
    print(f"Crime classes:    {category_to_id}")
    print(f"Color classes:    {color_to_id}")

    print("\nSplit sizes")
    print("-----------")
    print(f"Train:      {len(train_indices)} images")
    print(f"Validation: {len(validation_indices)} images")
    print(f"Test:       {len(test_indices)} images")

    print("\nCategory distribution")
    print("---------------------")
    print(
        annotations["category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    normalized_colors = (
        annotations["color_type"]
        .apply(normalize_color_label)
    )

    print("\nColor distribution")
    print("------------------")
    print(
        normalized_colors
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()