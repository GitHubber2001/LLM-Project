#!/usr/bin/env python3
"""Prepare Qwen instruction-tuning JSONL for all three PEFT conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.constants import LANGUAGE_COLUMNS, LANGUAGE_NAMES, SEED  # noqa: E402
from xed_project.io_utils import read_tsv, write_jsonl_atomic  # noqa: E402
from xed_project.prompting import INSTRUCTION  # noqa: E402


CONDITIONS = {
    "english_peft": ("en",),
    "turkish_peft": ("tr",),
    "bilingual_peft": ("en", "tr"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_records(
    source_rows: list[dict[str, str]],
    languages: tuple[str, ...],
    split: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in source_rows:
        gold_labels = json.loads(row["labels"])
        for language in languages:
            text = row[LANGUAGE_COLUMNS[language]]
            records.append(
                {
                    "alignment_id": row["alignment_id"],
                    "language": language,
                    "split": split,
                    "gold_labels": gold_labels,
                    "messages": [
                        {"role": "system", "content": INSTRUCTION},
                        {
                            "role": "user",
                            "content": f"{LANGUAGE_NAMES[language]} subtitle: {text}",
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(gold_labels, ensure_ascii=False),
                        },
                    ],
                }
            )
    random.Random(SEED).shuffle(records)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition", choices=tuple(CONDITIONS) + ("all",), default="all"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "peft")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions = CONDITIONS if args.condition == "all" else {
        args.condition: CONDITIONS[args.condition]
    }
    sources: dict[str, tuple[Path, list[dict[str, str]]]] = {}
    for split in ("train", "dev"):
        path = ROOT / "data" / "splits" / f"{split}.tsv"
        _, rows = read_tsv(path)
        sources[split] = (path, rows)

    for condition, languages in conditions.items():
        for split, (source_path, rows) in sources.items():
            records = build_records(rows, languages, split)
            output = args.output_dir / f"{condition}_{split}.jsonl"
            write_jsonl_atomic(output, records)
            expected = len(rows) * len(languages)
            if len(records) != expected:
                raise RuntimeError(f"PEFT record count mismatch: {len(records)} != {expected}")
            print(
                f"{condition} {split}: {len(records)} records -> {output} "
                f"(source SHA-256 {sha256(source_path)})"
            )


if __name__ == "__main__":
    main()
