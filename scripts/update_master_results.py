#!/usr/bin/env python3
"""Update one master-results row from a real completed metrics JSON file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.io_utils import write_csv_atomic  # noqa: E402


METRICS = (
    "micro_f1",
    "sample_jaccard",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "hamming_loss",
    "exact_match_ratio",
    "malformed_output_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--master", type=Path, default=ROOT / "results" / "master_results.csv"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or metrics.get("status") != "COMPLETED":
        raise ValueError("Metrics artifact is not marked COMPLETED")
    experiment_id = metrics.get("experiment_id")
    if not isinstance(experiment_id, str):
        raise ValueError("Metrics artifact lacks experiment_id")
    with args.master.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Master results has no header")
        fields = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    matches = [row for row in rows if row["experiment_id"] == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one master row for {experiment_id}, found {len(matches)}")
    row = matches[0]
    descriptors = {
        "model": metrics.get("model"),
        "condition": metrics.get("condition"),
        "evaluation_language": metrics.get("language"),
        "split": metrics.get("split"),
    }
    for field, observed in descriptors.items():
        if row[field] != observed:
            raise ValueError(
                f"Master/metrics descriptor mismatch for {field}: "
                f"{row[field]!r} != {observed!r}"
            )
    for metric in METRICS:
        value = metrics.get(metric)
        row[metric] = "" if value is None else str(value)
    row["status"] = "COMPLETED"
    metrics_path = args.metrics.resolve()
    row["metrics_file"] = str(metrics_path.relative_to(ROOT.resolve()))
    row["blocker"] = ""
    write_csv_atomic(args.master, fields, rows)
    print(f"updated {experiment_id} in {args.master}")


if __name__ == "__main__":
    main()
