#!/usr/bin/env python3
"""Evaluate machine-readable multi-label prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.constants import LABELS  # noqa: E402
from xed_project.io_utils import (  # noqa: E402
    read_jsonl,
    read_tsv,
    write_csv_atomic,
    write_json_atomic,
)
from xed_project.metrics import evaluate_label_sets  # noqa: E402


REQUIRED = {
    "alignment_id",
    "language",
    "model",
    "checkpoint_id",
    "condition",
    "split",
    "input_text",
    "gold_labels",
    "parsed_predicted_labels",
}


def validate_records(rows: list[dict[str, Any]]) -> dict[str, str]:
    if not rows:
        raise ValueError("Prediction file is empty")
    seen: set[tuple[str, str]] = set()
    descriptors: dict[str, set[str]] = {
        key: set()
        for key in ("language", "model", "checkpoint_id", "condition", "split")
    }
    for index, row in enumerate(rows, start=1):
        missing = REQUIRED - set(row)
        if missing:
            raise ValueError(f"Record {index} lacks fields: {sorted(missing)}")
        key = (str(row["alignment_id"]), str(row["language"]))
        if key in seen:
            raise ValueError(f"Duplicate prediction key: {key}")
        seen.add(key)
        for name in descriptors:
            descriptors[name].add(str(row[name]))
        for field in ("gold_labels", "parsed_predicted_labels"):
            values = row[field]
            if not isinstance(values, list) or not all(
                isinstance(label, str) and label in LABELS for label in values
            ):
                raise ValueError(f"Invalid {field} in record {index}")
    nonuniform = {key: value for key, value in descriptors.items() if len(value) != 1}
    if nonuniform:
        raise ValueError(f"Prediction descriptors are not uniform: {nonuniform}")
    result = {key: next(iter(value)) for key, value in descriptors.items()}
    for optional_descriptor in ("prompt_variant", "prompt_config_sha256"):
        values = [row.get(optional_descriptor) for row in rows]
        if any(value is not None and not isinstance(value, str) for value in values):
            raise ValueError(f"Invalid prediction {optional_descriptor}")
        unique = set(values)
        if len(unique) > 1:
            raise ValueError(f"Prediction {optional_descriptor} is not uniform")
        value = next(iter(unique))
        if isinstance(value, str) and value:
            result[optional_descriptor] = value
    return result


def validate_against_split(rows, descriptors):
    split = descriptors["split"]
    language = descriptors["language"]

    if split not in ("train", "dev", "test") or language not in ("en", "tr"):
        raise ValueError("Invalid split or language")

    _, split_rows = read_tsv(ROOT / "data" / "splits" / f"{split}.tsv")
    expected = {row["alignment_id"]: row for row in split_rows}

    if len(expected) != len(split_rows):
        raise ValueError("Duplicate IDs in reference split")

    prediction_ids = {row["alignment_id"] for row in rows}

    if len(prediction_ids) != len(rows):
        raise ValueError("Duplicate prediction IDs")

    if prediction_ids != set(expected):
        raise ValueError("Missing or unexpected prediction IDs")

    text_column = "english_text" if language == "en" else "turkish_text"

    for row in rows:
        source = expected[row["alignment_id"]]

        if row["gold_labels"] != json.loads(source["labels"]):
            raise ValueError("Gold labels do not match dataset")

        if row["input_text"] != source[text_column]:
            raise ValueError("Input text does not match dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    descriptors = validate_records(rows)
    validate_against_split(rows, descriptors)
    flags = None
    if all(isinstance(row.get("malformed"), bool) for row in rows):
        flags = [bool(row["malformed"]) for row in rows]
    evaluation = evaluate_label_sets(
        [row["gold_labels"] for row in rows],
        [row["parsed_predicted_labels"] for row in rows],
        flags,
    )
    overall = {
        "experiment_id": args.experiment_id,
        "status": "COMPLETED",
        **descriptors,
        "predictions_file": str(args.predictions),
        **evaluation.overall,
    }
    overall_path = args.output_dir / f"{args.experiment_id}_overall.json"
    per_label_path = args.output_dir / f"{args.experiment_id}_per_label.csv"
    write_json_atomic(overall_path, overall)
    write_csv_atomic(
        per_label_path,
        [
            "label",
            "precision",
            "recall",
            "f1",
            "support",
            "true_positive",
            "false_positive",
            "false_negative",
        ],
        evaluation.per_label,
    )
    print(json.dumps(overall, indent=2))
    print(f"per-label metrics: {per_label_path}")


if __name__ == "__main__":
    main()
