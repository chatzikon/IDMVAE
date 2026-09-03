#!/usr/bin/env python3
"""
Compare three VLMs on exactly the same UCA surveillance images.

Models
------
1. Qwen/Qwen2.5-VL-3B-Instruct
2. google/paligemma2-3b-mix-448
3. microsoft/Florence-2-large

The models are loaded ONE AT A TIME, so they do not need to fit in VRAM
simultaneously. After each model finishes, it is deleted and CUDA memory is
released before loading the next model.

The script:
  1) selects the same N images once (default: 100 from train),
  2) saves the selection as benchmark_manifest.jsonl,
  3) captions every selected image with each model,
  4) saves per-model JSONL results,
  5) creates comparison.csv,
  6) creates comparison.html with the image and all three captions side-by-side,
  7) creates summary.json with simple corpus statistics, runtime, and CUDA memory.

Important
---------
- No UCA label or previous caption is passed to any model.
- The default sampling is stratified by the existing `label` metadata so the
  100-image benchmark contains a broad range of scene categories. Labels are
  used ONLY to choose a varied benchmark sample and for later analysis.
- PaliGemma 2 is gated on Hugging Face. You must accept its license on the
  model page and be logged in (e.g. `huggingface-cli login`) before running it.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import logging
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


MODEL_SPECS = {
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "display_name": "Qwen2.5-VL-3B-Instruct",
    },
    # "paligemma": {
    #     "hf_id": "google/paligemma2-3b-mix-448",
    #     "display_name": "PaliGemma 2 3B Mix 448",
    # },
    # "florence": {
    #     "hf_id": "florence-community/Florence-2-large",
    #     "display_name": "Florence-2-large",
    # },
}

QWEN_PROMPT = """Describe the main visible content of this image in one simple factual English sentence of 8–15 words.

Mention only the most important people, clearly visible actions or posture, objects, vehicles, and immediate surroundings.

Describe only what can be directly seen in this single image.

Do not infer intentions, causes, crimes, identities, or events before or after the image.

Do not mention timestamps, dates, watermarks, camera labels, CCTV, surveillance footage, image quality, or text overlays.

Do not use uncertain phrases such as "possibly", "probably", "appears to be", or "seems to".

Prefer a simple subject–verb–object sentence.

Output only the caption."""

# Native captioning prompts recommended by the corresponding model families.
PALIGEMMA_PROMPT = "caption en"
FLORENCE_PROMPT = "<CAPTION>"

RISKY_CRIME_TERMS = {
    "robbery", "robber", "steal", "stealing", "stolen", "theft", "thief",
    "assault", "abuse", "arrest", "shoot", "shooting", "gunman", "burglary",
    "burglar", "vandalism", "vandal", "crime", "criminal"
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Caption the same UCA images with Qwen, PaliGemma 2 and Florence-2."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/chatziko/PycharmProjects/PythonProject/IDMVAE/UCA_image_dataset"),
        help="Dataset root containing train/validation/test.",
    )
    p.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Split from which benchmark images are selected.",
    )
    p.add_argument(
        "--metadata-name",
        default="metadata.jsonl",
        help="Metadata filename inside the selected split.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Benchmark output directory. Default: <data-root>/caption_model_comparison.",
    )
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sampling",
        choices=["stratified", "random", "first"],
        default="stratified",
        help="How to choose the benchmark images. Selection is saved and reused.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS),
        default=["qwen", "paligemma", "florence"],
        help="Models to run, in order.",
    )
    p.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    p.add_argument(
        "--device-map",
        default="auto",
        help='Passed to from_pretrained. Default: "auto".',
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum generated caption tokens.",
    )
    p.add_argument(
        "--overwrite-selection",
        action="store_true",
        help="Recreate the benchmark manifest instead of reusing it.",
    )
    p.add_argument(
        "--overwrite-model-results",
        action="store_true",
        help="Regenerate captions even if a model result already exists.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
    )
    args = p.parse_args()

    if args.num_images < 1:
        p.error("--num-images must be >= 1")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be >= 1")

    if args.output_dir is None:
        args.output_dir = args.data_root / "caption_model_comparison"

    return args


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "comparison.log", encoding="utf-8"),
        ],
    )


def choose_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_path_from_row(data_root: Path, split: str, row: dict[str, Any]) -> Path:
    raw = row.get("image")
    if not raw:
        raise KeyError("Metadata row has no 'image' field.")

    p = Path(str(raw))
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            data_root / p,
            data_root / split / p,
            data_root / split / "images" / p.name,
        ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not resolve image path: {raw}")


def normalize_caption(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Remove common wrappers.
    prefixes = ["caption:", "description:"]
    lower = text.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()

    return " ".join(text.split())


def create_stratified_sample(
    rows: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """
    Approximately balanced sample across `label`, while filling any unused
    quota from the remaining pool. The final order is shuffled deterministically.
    """
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        groups[str(row.get("label", "UNKNOWN"))].append(row)

    for group in groups.values():
        rng.shuffle(group)

    labels = sorted(groups)
    if not labels:
        return []

    base = n // len(labels)
    remainder = n % len(labels)

    selected: list[dict[str, Any]] = []
    selected_ids = set()

    for i, label in enumerate(labels):
        quota = base + (1 if i < remainder else 0)
        for row in groups[label][:quota]:
            key = str(row.get("image"))
            if key not in selected_ids:
                selected.append(row)
                selected_ids.add(key)

    # If a small class could not satisfy its quota, fill from all remaining rows.
    if len(selected) < n:
        remaining = [
            row for row in rows
            if str(row.get("image")) not in selected_ids
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    rng.shuffle(selected)
    return selected[:n]


def create_manifest(args) -> list[dict[str, Any]]:
    manifest_path = args.output_dir / "benchmark_manifest.jsonl"

    if manifest_path.is_file() and not args.overwrite_selection:
        manifest = read_jsonl(manifest_path)
        logging.info(
            "Reusing existing benchmark manifest with %d images: %s",
            len(manifest),
            manifest_path,
        )
        return manifest

    metadata_path = args.data_root / args.split / args.metadata_name
    rows = read_jsonl(metadata_path)

    # Keep only rows whose image can actually be resolved.
    valid = []
    for row in rows:
        try:
            image_path_from_row(args.data_root, args.split, row)
            valid.append(row)
        except FileNotFoundError:
            pass

    if len(valid) < args.num_images:
        raise RuntimeError(
            f"Requested {args.num_images} images, but only {len(valid)} valid images exist."
        )

    rng = random.Random(args.seed)

    if args.sampling == "first":
        selected = valid[: args.num_images]
    elif args.sampling == "random":
        selected = rng.sample(valid, args.num_images)
    else:
        selected = create_stratified_sample(valid, args.num_images, args.seed)

    manifest = []
    for idx, row in enumerate(selected, start=1):
        manifest.append({
            "sample_id": idx,
            "image": str(row["image"]),
            "label": row.get("label"),
            "annotation_id": row.get("annotation_id"),
            "video": row.get("video"),
            "segment_start": row.get("segment_start"),
            "segment_end": row.get("segment_end"),
            "selected_timestamp": row.get("selected_timestamp"),
        })

    write_jsonl(manifest_path, manifest)
    logging.info(
        "Created benchmark manifest: %d images | sampling=%s | seed=%d",
        len(manifest),
        args.sampling,
        args.seed,
    )

    counts = Counter(str(x.get("label", "UNKNOWN")) for x in manifest)
    logging.info("Benchmark label distribution: %s", dict(sorted(counts.items())))

    return manifest


def model_device(model) -> torch.device:
    try:
        dev = model.device
        if dev is not None and str(dev) != "meta":
            return torch.device(dev)
    except Exception:
        pass

    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_inputs(inputs, device: torch.device, dtype: torch.dtype | None = None):
    """
    Move tokenizer/processor output to device. Floating tensors may optionally
    be cast while token IDs remain integer.
    """
    result = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            result[key] = value
            continue
        value = value.to(device)
        if dtype is not None and value.is_floating_point():
            value = value.to(dtype=dtype)
        result[key] = value
    return result


def load_auto_multimodal_model(
    hf_id: str,
    dtype: torch.dtype,
    device_map: str,
    trust_remote_code: bool = False,
):
    """Compatibility loader for recent and older Transformers versions."""
    errors = []

    try:
        from transformers import AutoModelForMultimodalLM
        return AutoModelForMultimodalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(("AutoModelForMultimodalLM", exc))

    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(("AutoModelForImageTextToText", exc))

    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(("AutoModelForVision2Seq", exc))

    if trust_remote_code:
        try:
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                hf_id,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception as exc:
            errors.append(("AutoModelForCausalLM", exc))

    joined = " | ".join(f"{name}: {exc}" for name, exc in errors)
    raise RuntimeError(f"Could not load {hf_id}. {joined}")


def load_qwen(dtype, device_map):
    hf_id = MODEL_SPECS["qwen"]["hf_id"]
    processor = AutoProcessor.from_pretrained(hf_id)

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
    except Exception:
        model = load_auto_multimodal_model(hf_id, dtype, device_map)

    model.eval()

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if (
            processor.tokenizer.pad_token_id is None
            and processor.tokenizer.eos_token_id is not None
        ):
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    return model, processor


def load_paligemma(dtype, device_map):
    hf_id = MODEL_SPECS["paligemma"]["hf_id"]

    # PaliGemma 2 is gated: the user must have accepted the HF license.
    try:
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        processor = PaliGemmaProcessor.from_pretrained(hf_id)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(hf_id)
        model = load_auto_multimodal_model(hf_id, dtype, device_map)

    model.eval()
    return model, processor


def load_florence(dtype, device_map):
    hf_id = MODEL_SPECS["florence"]["hf_id"]
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model = load_auto_multimodal_model(
        hf_id,
        dtype,
        device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def caption_qwen(model, processor, image: Image.Image, max_new_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": QWEN_PROMPT},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    device = model_device(model)
    inputs = move_inputs(inputs, device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated = output[0, input_len:]
    text = processor.decode(generated, skip_special_tokens=True)
    return normalize_caption(text)


def caption_paligemma(model, processor, image: Image.Image, max_new_tokens: int) -> str:
    inputs = processor(
        text=PALIGEMMA_PROMPT,
        images=image,
        return_tensors="pt",
    )

    device = model_device(model)
    dtype = next(
        (p.dtype for p in model.parameters() if p.is_floating_point()),
        torch.float32,
    )
    inputs = move_inputs(inputs, device, dtype=dtype)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output[0, input_len:]
    text = processor.decode(generated, skip_special_tokens=True)
    return normalize_caption(text)


def caption_florence(model, processor, image: Image.Image, max_new_tokens: int) -> str:
    task = FLORENCE_PROMPT
    inputs = processor(
        text=task,
        images=image,
        return_tensors="pt",
    )

    device = model_device(model)
    dtype = next(
        (p.dtype for p in model.parameters() if p.is_floating_point()),
        torch.float32,
    )
    inputs = move_inputs(inputs, device, dtype=dtype)

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        output,
        skip_special_tokens=False,
    )[0]

    # Florence has task-specific post-processing. Fall back to ordinary decode
    # if the local Transformers/remote-code version returns an unexpected form.
    try:
        parsed = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )

        if isinstance(parsed, dict):
            value = parsed.get(task)
            if value is None and parsed:
                value = next(iter(parsed.values()))
            if isinstance(value, str):
                return normalize_caption(value)
            if value is not None:
                return normalize_caption(str(value))
    except Exception:
        pass

    decoded = processor.decode(output[0], skip_special_tokens=True)
    decoded = decoded.replace(task, "", 1)
    return normalize_caption(decoded)


def load_model_by_key(key: str, dtype, device_map):
    if key == "qwen":
        return load_qwen(dtype, device_map)
    if key == "paligemma":
        return load_paligemma(dtype, device_map)
    if key == "florence":
        return load_florence(dtype, device_map)
    raise KeyError(key)


def caption_with_model(
    key: str,
    model,
    processor,
    image: Image.Image,
    max_new_tokens: int,
) -> str:
    if key == "qwen":
        return caption_qwen(model, processor, image, max_new_tokens)
    if key == "paligemma":
        return caption_paligemma(model, processor, image, max_new_tokens)
    if key == "florence":
        return caption_florence(model, processor, image, max_new_tokens)
    raise KeyError(key)


def load_existing_model_results(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}

    rows = read_jsonl(path)
    return {
        int(row["sample_id"]): row
        for row in rows
        if "sample_id" in row
    }


def run_one_model(args, manifest, key: str) -> dict[str, Any]:
    spec = MODEL_SPECS[key]
    result_path = args.output_dir / f"captions_{key}.jsonl"

    if args.overwrite_model_results and result_path.exists():
        result_path.unlink()

    existing = load_existing_model_results(result_path)
    pending = [row for row in manifest if int(row["sample_id"]) not in existing]

    logging.info(
        "[%s] model=%s | completed=%d | pending=%d",
        key,
        spec["hf_id"],
        len(existing),
        len(pending),
    )

    if not pending:
        return {"reused": True}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_start = time.perf_counter()
    model, processor = load_model_by_key(key, choose_dtype(args.dtype), args.device_map)
    load_seconds = time.perf_counter() - load_start

    if torch.cuda.is_available():
        # reset_peak_memory_stats sets the current allocation as the new baseline,
        # so model weights still contribute to the reported total.
        torch.cuda.reset_peak_memory_stats()

    infer_start = time.perf_counter()
    new_count = 0
    errors = 0

    try:
        for i, row in enumerate(pending, start=1):
            image_path = image_path_from_row(args.data_root, args.split, row)

            try:
                with Image.open(image_path) as im:
                    image = im.convert("RGB")

                caption = caption_with_model(
                    key,
                    model,
                    processor,
                    image,
                    args.max_new_tokens,
                )
                image.close()

                result = {
                    **row,
                    "model_key": key,
                    "model_id": spec["hf_id"],
                    "caption": caption,
                    "status": "ok" if caption else "empty",
                }
            except torch.cuda.OutOfMemoryError:
                logging.exception("[%s] CUDA OOM at sample %s", key, row["sample_id"])
                raise
            except Exception as exc:
                logging.exception(
                    "[%s] Error on sample %s (%s)",
                    key,
                    row["sample_id"],
                    row["image"],
                )
                errors += 1
                result = {
                    **row,
                    "model_key": key,
                    "model_id": spec["hf_id"],
                    "caption": "",
                    "status": "error",
                    "error": str(exc),
                }

            append_jsonl(result_path, result)
            new_count += 1

            if i % args.log_every == 0 or i == len(pending):
                logging.info("[%s] %d/%d new samples", key, i, len(pending))

    finally:
        infer_seconds = time.perf_counter() - infer_start

        peak_allocated_gb = None
        peak_reserved_gb = None
        if torch.cuda.is_available():
            peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

        # Critical for a 16 GB GPU: unload before the next model.
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    return {
        "reused": False,
        "new_captions": new_count,
        "errors": errors,
        "load_seconds": load_seconds,
        "inference_seconds": infer_seconds,
        "seconds_per_image": infer_seconds / max(new_count, 1),
        "peak_cuda_allocated_gb": peak_allocated_gb,
        "peak_cuda_reserved_gb": peak_reserved_gb,
    }


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower())


def adjacent_repeat_rate(captions: list[str]) -> float:
    repeated = 0
    total = 0
    for caption in captions:
        toks = words(caption)
        for a, b in zip(toks, toks[1:]):
            total += 1
            if a == b:
                repeated += 1
    return repeated / total if total else 0.0


def risky_term_rate(captions: list[str]) -> float:
    if not captions:
        return 0.0
    count = 0
    for caption in captions:
        toks = set(words(caption))
        if toks & RISKY_CRIME_TERMS:
            count += 1
    return count / len(captions)


def distinct_n(captions: list[str], n: int) -> float:
    all_ngrams = []
    for caption in captions:
        toks = words(caption)
        all_ngrams.extend(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def corpus_stats(captions: list[str]) -> dict[str, Any]:
    captions = [c for c in captions if c]
    lengths = [len(words(c)) for c in captions]

    return {
        "num_captions": len(captions),
        "mean_words": statistics.mean(lengths) if lengths else 0.0,
        "median_words": statistics.median(lengths) if lengths else 0.0,
        "min_words": min(lengths) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
        "unique_caption_ratio": len(set(captions)) / len(captions) if captions else 0.0,
        "adjacent_repeat_rate": adjacent_repeat_rate(captions),
        "distinct_1": distinct_n(captions, 1),
        "distinct_2": distinct_n(captions, 2),
        "risky_crime_term_sentence_rate": risky_term_rate(captions),
    }


def merge_results(args, manifest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_model = {}
    for key in args.models:
        path = args.output_dir / f"captions_{key}.jsonl"
        if path.is_file():
            per_model[key] = load_existing_model_results(path)

    merged = []
    for row in manifest:
        sid = int(row["sample_id"])
        item = dict(row)
        for key in args.models:
            item[f"caption_{key}"] = per_model.get(key, {}).get(sid, {}).get("caption", "")
            item[f"status_{key}"] = per_model.get(key, {}).get(sid, {}).get("status", "missing")
        merged.append(item)

    # CSV
    csv_path = args.output_dir / "comparison.csv"
    fields = [
        "sample_id", "image", "label", "video",
        *[f"caption_{key}" for key in args.models],
        *[f"status_{key}" for key in args.models],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    # Summary stats
    summary = {
        "split": args.split,
        "num_images": len(manifest),
        "sampling": args.sampling,
        "seed": args.seed,
        "models": {},
        "pairwise_caption_similarity": {},
    }

    for key in args.models:
        caps = [row.get(f"caption_{key}", "") for row in merged]
        summary["models"][key] = {
            "model_id": MODEL_SPECS[key]["hf_id"],
            **corpus_stats(caps),
        }

    # Simple lexical agreement. This is NOT an image-caption quality metric;
    # it only shows how similarly the models phrase the same images.
    for i, a in enumerate(args.models):
        for b in args.models[i+1:]:
            sims = []
            for row in merged:
                ca = row.get(f"caption_{a}", "")
                cb = row.get(f"caption_{b}", "")
                if ca and cb:
                    sims.append(SequenceMatcher(None, ca.lower(), cb.lower()).ratio())
            summary["pairwise_caption_similarity"][f"{a}__{b}"] = (
                statistics.mean(sims) if sims else None
            )

    return merged, summary


def write_html(args, merged: list[dict[str, Any]]) -> Path:
    html_path = args.output_dir / "comparison.html"

    model_headers = "".join(
        f"<th>{html.escape(MODEL_SPECS[k]['display_name'])}</th>"
        for k in args.models
    )

    rows_html = []
    for row in merged:
        img_path = image_path_from_row(args.data_root, args.split, row)
        rel_img = os.path.relpath(img_path, start=args.output_dir)

        captions = "".join(
            "<td>" + html.escape(row.get(f"caption_{k}", "")) + "</td>"
            for k in args.models
        )

        rows_html.append(
            f"""
            <tr>
              <td>{int(row['sample_id'])}</td>
              <td><img src="{html.escape(rel_img)}" loading="lazy"></td>
              <td>{html.escape(str(row.get('label', '')))}</td>
              {captions}
            </tr>
            """
        )

    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>UCA Caption Model Comparison</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 20px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
}}
th, td {{
    border: 1px solid #bbb;
    padding: 8px;
    vertical-align: top;
    text-align: left;
}}
th {{
    position: sticky;
    top: 0;
    background: white;
}}
img {{
    max-width: 320px;
    max-height: 220px;
    object-fit: contain;
}}
tr:nth-child(even) {{
    background: #f7f7f7;
}}
.caption-note {{
    margin-bottom: 16px;
    max-width: 1000px;
}}
</style>
</head>
<body>
<h1>UCA Caption Model Comparison</h1>
<div class="caption-note">
<p>
Same {len(merged)} images for every model. The dataset label is displayed only
for human analysis; it was not passed to any VLM.
</p>
<p>
Qwen uses the constrained surveillance-image instruction. PaliGemma and
Florence use their native short-caption tasks (<code>caption en</code> and
<code>&lt;CAPTION&gt;</code>) because those checkpoints are explicitly trained
around task prefixes.
</p>
</div>
<table>
<thead>
<tr>
<th>#</th>
<th>Image</th>
<th>Dataset label<br>(not model input)</th>
{model_headers}
</tr>
</thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""

    html_path.write_text(document, encoding="utf-8")
    return html_path


def main() -> int:
    args = parse_args()
    configure_logging(args.output_dir)

    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)

    logging.info("Data root: %s", args.data_root)
    logging.info("Output: %s", args.output_dir)
    logging.info("Models will be loaded sequentially: %s", args.models)

    manifest = create_manifest(args)

    runtime = {}
    for key in args.models:
        logging.info("=" * 70)
        logging.info("Starting %s", MODEL_SPECS[key]["display_name"])
        runtime[key] = run_one_model(args, manifest, key)

    merged, summary = merge_results(args, manifest)
    summary["runtime"] = runtime

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    html_path = write_html(args, merged)

    logging.info("Comparison CSV:  %s", args.output_dir / "comparison.csv")
    logging.info("Comparison HTML: %s", html_path)
    logging.info("Summary:         %s", summary_path)

    if torch.cuda.is_available():
        logging.info(
            "Final CUDA allocated/reserved: %.2f / %.2f GB",
            torch.cuda.memory_allocated() / (1024 ** 3),
            torch.cuda.memory_reserved() / (1024 ** 3),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())