import json
from pathlib import Path


DATA_ROOT = Path(
    "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/"
    "archive/UCA_image_dataset"
)

# Change these if your generated caption files have different names/locations.
CAPTION_FILES = {
    "train": DATA_ROOT / "train" / "captions_vlm.jsonl",
    "validation": DATA_ROOT / "validation" / "captions_vlm.jsonl",
    "test": DATA_ROOT / "test" / "captions_vlm.jsonl",
}

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT_VERSION = "uca_image_caption_v2"


def read_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}"
                ) from e

    return rows


def image_key(value):
    """
    Match using filename.

    If filenames are globally unique in your extracted dataset,
    this is sufficient and robust to differences such as:
        train/images/a.jpg
        /absolute/path/train/images/a.jpg
    """
    return Path(value).name


for split_name in ("train", "validation", "test"):

    split_dir = DATA_ROOT / split_name

    metadata_path = split_dir / "metadata.jsonl"
    captions_path = CAPTION_FILES[split_name]

    metadata = read_jsonl(metadata_path)
    caption_rows = read_jsonl(captions_path)

    # -------------------------------------------------------------
    # Construct image -> generated caption mapping
    # -------------------------------------------------------------

    caption_by_image = {}

    for row in caption_rows:

        if "image" not in row:
            raise ValueError(
                f"Caption record does not contain 'image': {row}"
            )

        key = image_key(row["image"])

        if key in caption_by_image:
            raise ValueError(
                f"Duplicate generated caption for image: {key}"
            )

        # Support both output conventions.
        caption = row.get("generated_caption")

        if caption is None:
            caption = row.get("caption")

        if caption is None:
            raise ValueError(
                f"No caption field found for image {key}"
            )

        caption = str(caption).strip()

        if not caption:
            raise ValueError(
                f"Empty generated caption for image {key}"
            )

        caption_by_image[key] = caption

    # -------------------------------------------------------------
    # Insert caption into metadata
    # -------------------------------------------------------------

    matched = set()

    for record in metadata:

        key = image_key(record["image"])

        if key not in caption_by_image:
            raise ValueError(
                f"No generated caption found for metadata image: {key}"
            )

        record["caption"] = caption_by_image[key]
        record["caption_model"] = MODEL_NAME
        record["caption_prompt_version"] = PROMPT_VERSION

        matched.add(key)

    # -------------------------------------------------------------
    # Strict consistency checks
    # -------------------------------------------------------------

    unused_captions = set(caption_by_image) - matched

    if unused_captions:
        raise ValueError(
            f"{len(unused_captions)} captions do not have a corresponding "
            f"metadata record. Example: {next(iter(unused_captions))}"
        )

    if len(metadata) != len(caption_by_image):
        raise ValueError(
            f"Mismatch for {split_name}: "
            f"{len(metadata)} metadata rows vs "
            f"{len(caption_by_image)} captions."
        )

    # -------------------------------------------------------------
    # Keep original metadata and write a new captioned version
    # -------------------------------------------------------------

    output_path = split_dir / "metadata_captioned.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for record in metadata:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(
        f"{split_name:<10}: "
        f"{len(metadata):>6} records -> {output_path}"
    )

print("\nAll caption files merged successfully.")