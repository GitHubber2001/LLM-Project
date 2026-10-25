#!/usr/bin/env python3
"""Create comparison candidates from two real prediction JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.io_utils import read_jsonl, write_csv_atomic  # noqa: E402


FIELDS = [
    "alignment_id",
    "baseline_language",
    "comparison_language",
    "baseline_condition",
    "comparison_condition",
    "input_text",
    "gold_labels",
    "baseline_labels",
    "comparison_labels",
    "baseline_exact",
    "comparison_exact",
    "baseline_malformed",
    "comparison_malformed",
    "category",
    "review_category",
    "possible_alignment_limitation",
    "possible_projection_label_limitation",
    "review_notes",
]


def by_alignment(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        alignment_id = row.get("alignment_id")
        if not isinstance(alignment_id, str) or alignment_id in result:
            raise ValueError(f"Invalid/duplicate alignment ID: {alignment_id!r}")
        result[alignment_id] = row
    return result


def category(gold: set[str], baseline: set[str], comparison: set[str]) -> str:
    baseline_exact = baseline == gold
    comparison_exact = comparison == gold
    if not baseline_exact and comparison_exact:
        return "baseline_wrong_comparison_correct"
    if baseline_exact and not comparison_exact:
        return "baseline_correct_comparison_wrong"
    if baseline_exact and comparison_exact:
        return "both_exact"
    baseline_jaccard = len(gold & baseline) / len(gold | baseline) if gold | baseline else 1.0
    comparison_jaccard = len(gold & comparison) / len(gold | comparison) if gold | comparison else 1.0
    if comparison_jaccard > baseline_jaccard:
        return "partial_match_improved"
    if comparison_jaccard < baseline_jaccard:
        return "partial_match_worsened"
    return "both_inexact_same_jaccard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = by_alignment(read_jsonl(args.baseline))
    comparison = by_alignment(read_jsonl(args.comparison))
    shared = sorted(set(baseline) & set(comparison))
    if not shared:
        raise ValueError("Prediction files share no alignment IDs")
    output: list[dict[str, object]] = []
    for alignment_id in shared:
        left, right = baseline[alignment_id], comparison[alignment_id]
        if left.get("gold_labels") != right.get("gold_labels"):
            raise ValueError(f"Gold-label mismatch: {alignment_id}")
        gold = set(left["gold_labels"])
        left_pred = set(left["parsed_predicted_labels"])
        right_pred = set(right["parsed_predicted_labels"])
        item_category = category(gold, left_pred, right_pred)
        if item_category == "both_exact":
            continue
        output.append(
            {
                "alignment_id": alignment_id,
                "baseline_language": left.get("language", ""),
                "comparison_language": right.get("language", ""),
                "baseline_condition": left.get("condition", ""),
                "comparison_condition": right.get("condition", ""),
                "input_text": right.get("input_text", ""),
                "gold_labels": json.dumps(left["gold_labels"]),
                "baseline_labels": json.dumps(left["parsed_predicted_labels"]),
                "comparison_labels": json.dumps(right["parsed_predicted_labels"]),
                "baseline_exact": left_pred == gold,
                "comparison_exact": right_pred == gold,
                "baseline_malformed": bool(left.get("malformed", False)),
                "comparison_malformed": bool(right.get("malformed", False)),
                "category": item_category,
                "review_category": "unreviewed",
                "possible_alignment_limitation": "",
                "possible_projection_label_limitation": "",
                "review_notes": "",
            }
        )
    write_csv_atomic(args.output, FIELDS, output)
    print(f"error candidates: {len(output)} -> {args.output}")
    print(
        "Causal categories remain unreviewed; inspect the aligned pair and projected "
        "reference before marking model, alignment, or projection limitations."
    )


if __name__ == "__main__":
    main()
