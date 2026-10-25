#!/usr/bin/env python3
"""Select five fixed aligned demonstrations using TRAIN data only.

Selection greedily maximizes newly covered emotion labels, then gives priority
to rarer newly covered labels. Remaining deterministic tie-breakers prefer
shorter bilingual examples and finally the alignment ID. No DEV/TEST rows or
model outputs are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.constants import LABELS, SEED  # noqa: E402
from xed_project.io_utils import read_tsv, write_json_atomic  # noqa: E402


DEFAULT_TRAIN = ROOT / "data" / "splits" / "train.tsv"
DEFAULT_OUTPUT = ROOT / "configs" / "five_shot_examples.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_examples(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    candidates: list[tuple[dict[str, str], tuple[str, ...]]] = []
    frequencies: Counter[str] = Counter()
    for row in rows:
        labels = json.loads(row["labels"])
        if not isinstance(labels, list) or not labels or any(
            label not in LABELS for label in labels
        ):
            raise ValueError(f"Invalid TRAIN labels: {row['alignment_id']}")
        label_tuple = tuple(labels)
        frequencies.update(label_tuple)
        if (
            8 <= len(row["english_text"].strip()) <= 160
            and 8 <= len(row["turkish_text"].strip()) <= 160
            and len(label_tuple) <= 2
        ):
            candidates.append((row, label_tuple))
    if len(candidates) < 5:
        raise ValueError("Fewer than five length-eligible TRAIN examples")

    selected: list[tuple[dict[str, str], tuple[str, ...]]] = []
    covered: set[str] = set()
    remaining = candidates.copy()
    for _ in range(5):
        def selection_key(item: tuple[dict[str, str], tuple[str, ...]]):
            row, labels = item
            newly_covered = set(labels) - covered
            rare_weight = sum(1.0 / frequencies[label] for label in newly_covered)
            total_length = len(row["english_text"]) + len(row["turkish_text"])
            return (
                -len(newly_covered),
                -rare_weight,
                total_length,
                row["alignment_id"],
            )

        chosen = min(remaining, key=selection_key)
        selected.append(chosen)
        covered.update(chosen[1])
        chosen_english = chosen[0]["english_text"]
        chosen_turkish = chosen[0]["turkish_text"]
        remaining = [
            item
            for item in remaining
            if item[0]["english_text"] != chosen_english
            and item[0]["turkish_text"] != chosen_turkish
        ]

    if covered != set(LABELS):
        raise ValueError(
            "Five-example greedy selection did not cover all labels: "
            f"missing={sorted(set(LABELS) - covered)}"
        )

    return [
        {
            "selection_rank": rank,
            "alignment_id": row["alignment_id"],
            "english_text": row["english_text"],
            "turkish_text": row["turkish_text"],
            "gold_labels": list(labels),
        }
        for rank, (row, labels) in enumerate(selected, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, rows = read_tsv(args.train)
    examples = select_examples(rows)
    artifact = {
        "version": 1,
        "seed": SEED,
        "source_split": "train",
        "source_file": str(args.train.relative_to(ROOT)),
        "source_sha256": sha256(args.train),
        "selection_procedure": (
            "TRAIN-only deterministic greedy coverage: maximize the number of "
            "new emotion labels, then inverse-frequency weight of new labels, "
            "then prefer shorter combined English/Turkish text and alignment ID; "
            "candidates contain at most two labels and selected examples cannot "
            "repeat exact English or Turkish text"
        ),
        "length_filter_characters_per_language": {"minimum": 8, "maximum": 160},
        "maximum_labels_per_example": 2,
        "covered_labels": list(LABELS),
        "examples": examples,
    }
    write_json_atomic(args.output, artifact)
    print(f"selected examples: {len(examples)}")
    print("covered labels:", ", ".join(sorted({label for e in examples for label in e["gold_labels"]})))
    for example in examples:
        print(
            f"{example['selection_rank']}: {example['alignment_id']} "
            f"{example['gold_labels']}"
        )
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
