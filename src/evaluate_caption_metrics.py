#!/usr/bin/env python3
"""
Evaluate IDMVAE image-caption predictions with:
BLEU-1/2/3/4, ROUGE-L, METEOR, CIDEr, and SPICE.

Typical IDMVAE input files:
    references_epoch5.json
    A_zimg_wprior_epoch5.json
    A_zimg_wprior_epoch10.json
    A_zimg_wprior_epoch20.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

try:
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pycocoevalcap\n"
        "Install it with:\n"
        "    pip install pycocoevalcap\n"
    ) from exc


CaptionDict = Dict[str, str]
ReferenceDict = Dict[str, List[str]]


def clean_caption(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(path: str | Path) -> CaptionDict:
    data = load_json(path)

    if isinstance(data, dict):
        out: CaptionDict = {}
        for key, value in data.items():
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(
                        f"{key}: predictions must contain exactly one caption."
                    )
                value = value[0]
            out[str(key)] = clean_caption(value)
        return out

    if isinstance(data, list):
        out = {}
        for item in data:
            if "image_id" not in item or "caption" not in item:
                raise ValueError(
                    "List-format predictions require image_id and caption."
                )
            out[str(item["image_id"])] = clean_caption(item["caption"])
        return out

    raise ValueError(f"Unsupported predictions format: {type(data).__name__}")


def load_references(path: str | Path) -> ReferenceDict:
    data = load_json(path)

    if isinstance(data, dict):
        out: ReferenceDict = {}
        for key, value in data.items():
            refs = value if isinstance(value, list) else [value]
            refs = [clean_caption(x) for x in refs if clean_caption(x)]
            if not refs:
                raise ValueError(f"{key}: no valid reference caption.")
            out[str(key)] = refs
        return out

    if isinstance(data, list):
        out: ReferenceDict = {}
        for item in data:
            if "image_id" not in item or "caption" not in item:
                raise ValueError(
                    "List-format references require image_id and caption."
                )
            key = str(item["image_id"])
            caption = clean_caption(item["caption"])
            if caption:
                out.setdefault(key, []).append(caption)
        return out

    raise ValueError(f"Unsupported references format: {type(data).__name__}")


def checked_keys(
    references: Mapping[str, Sequence[str]],
    predictions: Mapping[str, str],
    allow_partial: bool,
) -> List[str]:
    ref_keys = set(references)
    pred_keys = set(predictions)

    missing = sorted(ref_keys - pred_keys)
    extra = sorted(pred_keys - ref_keys)

    if (missing or extra) and not allow_partial:
        raise ValueError(
            "Prediction/reference keys do not match.\n"
            f"References: {len(ref_keys)}\n"
            f"Predictions: {len(pred_keys)}\n"
            f"Missing predictions: {len(missing)}\n"
            f"Extra predictions: {len(extra)}\n"
            "Use --allow-partial only if this mismatch is intentional."
        )

    keys = sorted(ref_keys & pred_keys)
    if not keys:
        raise ValueError("No common image keys between references and predictions.")
    return keys


def compute_caption_metrics(
    references: ReferenceDict,
    predictions: CaptionDict,
    include_spice: bool = True,
    allow_partial: bool = False,
) -> Dict[str, float]:

    keys = checked_keys(references, predictions, allow_partial)

    # PTBTokenizer expects COCO-style annotation dictionaries.
    gts_raw = {
        key: [{"caption": ref} for ref in references[key]]
        for key in keys
    }
    res_raw = {
        key: [{"caption": predictions[key]}]
        for key in keys
    }

    print(f"Evaluating {len(keys)} image-caption pairs")
    print("Tokenizing with PTBTokenizer...")

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts_raw)
    res = tokenizer.tokenize(res_raw)

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(), "ROUGE-L"),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
    ]

    if include_spice:
        scorers.append((Spice(), "SPICE"))

    metrics: Dict[str, float] = {}

    for scorer, names in scorers:
        print(f"Computing {scorer.method()}...")
        score, _ = scorer.compute_score(gts, res)

        if isinstance(names, list):
            for name, value in zip(names, score):
                metrics[name] = float(value)
        else:
            metrics[names] = float(score)

    return metrics


def print_metrics(name: str, metrics: Mapping[str, float]) -> None:
    order = [
        "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
        "ROUGE-L", "METEOR", "SPICE", "CIDEr",
    ]
    print("\n" + "=" * 64)
    print(name)
    print("=" * 64)
    for metric in order:
        if metric in metrics:
            print(f"{metric:10s}: {metrics[metric]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True)
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="One or more A_zimg_wprior_epoch*.json files.",
    )
    parser.add_argument("--output-dir", default="caption_metrics")
    parser.add_argument("--no-spice", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    references = load_references(args.references)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, float]] = {}

    for pred_path_str in args.predictions:
        pred_path = Path(pred_path_str)
        predictions = load_predictions(pred_path)

        metrics = compute_caption_metrics(
            references,
            predictions,
            include_spice=not args.no_spice,
            allow_partial=args.allow_partial,
        )

        name = pred_path.stem
        all_results[name] = metrics
        print_metrics(name, metrics)

        with (output_dir / f"{name}_metrics.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(metrics, f, indent=4)

    metric_order = [
        "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
        "ROUGE-L", "METEOR", "SPICE", "CIDEr",
    ]

    with (output_dir / "all_caption_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(all_results, f, indent=4)

    with (output_dir / "all_caption_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["checkpoint"] + metric_order
        )
        writer.writeheader()
        for checkpoint, metrics in all_results.items():
            row = {"checkpoint": checkpoint}
            row.update(metrics)
            writer.writerow(row)

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()