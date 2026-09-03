#!/usr/bin/env python3
"""
Generate image-grounded captions for the UCA/UCF-Crime still-image dataset.

Expected structure
------------------
DATA_ROOT/
├── train/
│   ├── images/
│   └── metadata.jsonl
├── validation/
│   ├── images/
│   └── metadata.jsonl
└── test/
    ├── images/
    └── metadata.jsonl

Important: the VLM sees ONLY the image and the instruction. It never receives
UCA crime labels, source-folder names, or old captions.

Workflow
--------
1. Generate captions:
   python caption_UCA_images.py --mode caption

   Output per split: captions_vlm.jsonl

2. Optional grounding verification/rewrite:
   python caption_UCA_images.py --mode verify

   Input per split:  captions_vlm.jsonl
   Output per split: captions_verified.jsonl

The script is resume-safe and supports batched Hugging Face VLM inference.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from transformers import AutoProcessor


DEFAULT_CAPTION_PROMPT = """Describe the main visible content of this image in one simple factual English sentence of 8–15 words.

Mention only the most important people, clearly visible actions or posture, objects, vehicles, and immediate surroundings.

Describe only what can be directly seen in this single image.

Do not infer intentions, causes, crimes, identities, or events before or after the image.

Do not mention timestamps, dates, watermarks, camera labels, CCTV, surveillance footage, image quality, or text overlays.

Do not use uncertain phrases such as "possibly", "probably", "appears to be", or "seems to".

Prefer a simple subject–verb–object sentence.

Output only the caption."""

VERIFY_PROMPT_TEMPLATE = """Check whether the proposed caption is fully supported by this single surveillance image.

Proposed caption:
\"{caption}\"

Rules:
- Every factual claim must be directly visible in the image.
- Do not use information about what happened before or after the still image.
- Do not infer intentions, motives, identities, or crime labels.
- If the caption is fully supported, keep it unchanged.
- If any claim is unsupported, rewrite it as one concise factual sentence using only visible information.

Return EXACTLY one JSON object and nothing else:
{{\"supported\": true, \"caption\": \"caption text\"}}
or
{{\"supported\": false, \"caption\": \"rewritten caption text\"}}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or verify image-grounded captions for UCA still images."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/chatziko/PycharmProjects/PythonProject/IDMVAE/UCA_image_dataset"),
        help="Root containing train/, validation/, and test/.",
    )
    parser.add_argument(
        "--mode", choices=["caption", "verify"], default="caption"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face VLM checkpoint.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        default=["train", "validation", "test"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Images per batch. Reduce this if VRAM is insufficient.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum images PER SPLIT for testing.",
    )
    parser.add_argument(
        "--caption-input-name", type=str, default="metadata.jsonl"
    )
    parser.add_argument(
        "--caption-output-name", type=str, default="captions_vlm.jsonl"
    )
    parser.add_argument(
        "--verify-output-name", type=str, default="captions_verified.jsonl"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete mode-specific output before processing.",
    )
    parser.add_argument(
        "--device-map", type=str, default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--caption-prompt", type=str, default=DEFAULT_CAPTION_PROMPT
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--trust-remote-code", action="store_true", default=True
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )

    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be >= 1")
    return args


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def choose_dtype(name: str):
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_model(model_name: str, dtype, device_map: str, trust_remote_code: bool):
    """Load a multimodal generation model with compatibility fallbacks."""
    errors = []

    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(("AutoModelForImageTextToText", exc))

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(("Qwen2_5_VLForConditionalGeneration", exc))

    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        errors.append(("AutoModelForVision2Seq", exc))

    message = " | ".join(f"{name}: {exc}" for name, exc in errors)
    raise RuntimeError(f"Could not load model {model_name!r}. {message}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
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


def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def completed_images(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("image"):
                result.add(str(row["image"]))
    return result


def chunks(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def image_path_from_row(data_root: Path, split: str, row: dict[str, Any]) -> Path:
    raw = row.get("image")
    if not raw:
        raise KeyError("Metadata row has no 'image' field")
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


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def build_chat_text(processor, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def move_inputs(inputs, model):
    try:
        device = model.device
        if device is not None and str(device) != "meta":
            return inputs.to(device)
    except Exception:
        pass

    if hasattr(model, "hf_device_map"):
        for dev in model.hf_device_map.values():
            if isinstance(dev, int):
                return inputs.to(torch.device(f"cuda:{dev}"))
            if isinstance(dev, str) and dev.startswith("cuda"):
                return inputs.to(torch.device(dev))
    return inputs


def normalize_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return " ".join(text.split())


def generate_batch(model, processor, images, prompts, max_new_tokens: int) -> list[str]:
    texts = [build_chat_text(processor, p) for p in prompts]
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs(inputs, model)
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated = generated[:, input_length:]
    outputs = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [normalize_caption(x) for x in outputs]


def extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def caption_split(args, split: str, model, processor) -> None:
    split_dir = args.data_root / split
    input_path = split_dir / args.caption_input_name
    output_path = split_dir / args.caption_output_name

    if args.overwrite and output_path.exists():
        output_path.unlink()

    rows = read_jsonl(input_path)
    done = completed_images(output_path)
    pending = [r for r in rows if str(r.get("image")) not in done]
    if args.max_images is not None:
        pending = pending[: args.max_images]

    logging.info(
        "[%s] rows=%d | already_done=%d | pending=%d",
        split, len(rows), len(done), len(pending)
    )

    processed = 0
    with output_path.open("a", encoding="utf-8") as out:
        for batch_rows in chunks(pending, args.batch_size):
            images = []
            valid_rows = []

            for row in batch_rows:
                try:
                    image_path = image_path_from_row(args.data_root, split, row)
                    images.append(open_rgb(image_path))
                    valid_rows.append(row)
                except Exception as exc:
                    logging.exception("[%s] Failed to load %s", split, row.get("image"))
                    result = dict(row)
                    result.update({
                        "generated_caption": "",
                        "caption_status": "image_load_error",
                        "caption_error": str(exc),
                        "caption_model": args.model,
                    })
                    append_jsonl(out, result)

            if not valid_rows:
                continue

            try:
                captions = generate_batch(
                    model,
                    processor,
                    images,
                    [args.caption_prompt] * len(images),
                    args.max_new_tokens,
                )
            except torch.cuda.OutOfMemoryError:
                logging.error(
                    "CUDA OOM with batch size %d. Reduce --batch-size and rerun; completed rows will be skipped.",
                    len(valid_rows),
                )
                raise
            finally:
                for image in images:
                    image.close()

            for row, caption in zip(valid_rows, captions):
                result = dict(row)
                result.update({
                    "generated_caption": caption,
                    "caption_status": "generated" if caption else "empty_generation",
                    "caption_model": args.model,
                    "caption_prompt_version": "uca_image_only_v1",
                })
                append_jsonl(out, result)
                processed += 1
                if processed % args.log_every == 0:
                    logging.info("[%s] processed %d/%d", split, processed, len(pending))

    logging.info("[%s] caption generation complete: %d new rows", split, processed)


def verify_split(args, split: str, model, processor) -> None:
    split_dir = args.data_root / split
    input_path = split_dir / args.caption_output_name
    output_path = split_dir / args.verify_output_name

    if args.overwrite and output_path.exists():
        output_path.unlink()

    rows = read_jsonl(input_path)
    done = completed_images(output_path)
    pending = [
        r for r in rows
        if str(r.get("image")) not in done and r.get("generated_caption")
    ]
    if args.max_images is not None:
        pending = pending[: args.max_images]

    logging.info(
        "[%s verify] rows=%d | already_done=%d | pending=%d",
        split, len(rows), len(done), len(pending)
    )

    processed = 0
    rewritten = 0
    parse_errors = 0

    with output_path.open("a", encoding="utf-8") as out:
        for batch_rows in chunks(pending, args.batch_size):
            images = []
            valid_rows = []
            prompts = []

            for row in batch_rows:
                try:
                    image_path = image_path_from_row(args.data_root, split, row)
                    images.append(open_rgb(image_path))
                    valid_rows.append(row)
                    safe_caption = str(row["generated_caption"]).replace('"', '\\"')
                    prompts.append(VERIFY_PROMPT_TEMPLATE.format(caption=safe_caption))
                except Exception as exc:
                    logging.exception("[%s verify] Failed to load %s", split, row.get("image"))
                    result = dict(row)
                    result.update({
                        "verified_caption": row.get("generated_caption", ""),
                        "verification_status": "image_load_error",
                        "verification_error": str(exc),
                        "verification_model": args.model,
                    })
                    append_jsonl(out, result)

            if not valid_rows:
                continue

            try:
                raw_outputs = generate_batch(
                    model,
                    processor,
                    images,
                    prompts,
                    max(args.max_new_tokens, 96),
                )
            except torch.cuda.OutOfMemoryError:
                logging.error(
                    "CUDA OOM in verification with batch size %d. Reduce --batch-size and resume.",
                    len(valid_rows),
                )
                raise
            finally:
                for image in images:
                    image.close()

            for row, raw in zip(valid_rows, raw_outputs):
                parsed = extract_json_object(raw)
                if parsed is None:
                    supported = None
                    final_caption = str(row["generated_caption"])
                    status = "parse_error"
                    parse_errors += 1
                else:
                    supported = bool(parsed.get("supported"))
                    final_caption = normalize_caption(
                        str(parsed.get("caption") or row["generated_caption"])
                    )
                    status = "passed" if supported else "rewritten"
                    if not supported:
                        rewritten += 1

                result = dict(row)
                result.update({
                    "verified_caption": final_caption,
                    "verification_supported": supported,
                    "verification_status": status,
                    "verification_model": args.model,
                    "verification_raw_output": raw,
                    "verification_prompt_version": "uca_grounding_verify_v1",
                })
                append_jsonl(out, result)
                processed += 1
                if processed % args.log_every == 0:
                    logging.info(
                        "[%s verify] processed %d/%d | rewritten=%d | parse_errors=%d",
                        split, processed, len(pending), rewritten, parse_errors
                    )

    logging.info(
        "[%s verify] complete | processed=%d | rewritten=%d | parse_errors=%d",
        split, processed, rewritten, parse_errors
    )


def main() -> int:
    args = parse_args()
    configure_logging()

    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)

    dtype = choose_dtype(args.dtype)
    logging.info("Loading processor: %s", args.model)
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if (
            processor.tokenizer.pad_token_id is None
            and processor.tokenizer.eos_token_id is not None
        ):
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    logging.info("Loading model: %s | dtype=%s", args.model, dtype)
    model = load_model(
        args.model, dtype, args.device_map, args.trust_remote_code
    )
    model.eval()

    for split in args.splits:
        if args.mode == "caption":
            caption_split(args, split, model, processor)
        else:
            verify_split(args, split, model, processor)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info("All requested splits completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())