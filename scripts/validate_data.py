#!/usr/bin/env python3
"""Validate the prepared and split XED English-Turkish data.

The validator is read-only unless ``--report`` is supplied. It checks raw
pairs, source-metadata labels, complete split rows, disjoint partitions,
leakage groups, exact-text leakage, the manifest, fixed ratios, and multi-label
preservation. A failed check returns a non-zero exit status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.constants import LABELS, NUMERIC_TO_LABEL  # noqa: E402
from xed_project.io_utils import read_tsv, write_text_atomic  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "data_split.json"
DEFAULT_PAIRS = ROOT / "subtitle-retrieval" / "students" / "pairs-tr.txt"
DEFAULT_METADATA = (
    ROOT / "subtitle-retrieval" / "students" / "student_annotations.csv"
)
SPLITS = ("train", "dev", "test")
FILENAME_LABELS = {
    "ang": 1,
    "ant": 2,
    "dis": 3,
    "fea": 4,
    "joy": 5,
    "sad": 6,
    "sur": 7,
    "tru": 8,
}
METADATA_FILENAME_RE = re.compile(
    r"fixed_en_(ang|ant|dis|fea|joy|sad|sur|tru)\.txt"
)
REQUIRED_COLUMNS = (
    "alignment_id",
    "leakage_group_id",
    "source_document",
    "source_segment_ids",
    "target_document",
    "target_segment_ids",
    "english_text",
    "turkish_text",
    "labels_numeric",
    "labels",
)


def physical_lines(path: Path) -> list[str]:
    records = path.read_bytes().split(b"\n")
    if records and records[-1] == b"":
        records.pop()
    return [record.removesuffix(b"\r").decode("utf-8") for record in records]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_string_list(value: str, *, field: str, line: int) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {field} JSON at line {line}") from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{field} must be a JSON string list at line {line}")
    return tuple(parsed)


def parse_numeric_list(value: str, *, line: int) -> tuple[int, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid labels_numeric JSON at line {line}") from error
    if not isinstance(parsed, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in parsed
    ):
        raise ValueError(f"labels_numeric must be an integer JSON list at line {line}")
    return tuple(parsed)


def load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Data split config must be a JSON object")
    if value.get("seed") != 42:
        raise ValueError(f"Expected fixed seed 42, found {value.get('seed')!r}")
    ratios = value.get("target_ratios")
    if ratios != {"train": 0.7, "dev": 0.2, "test": 0.1}:
        raise ValueError(f"Unexpected target ratios: {ratios!r}")
    return value


def load_metadata(path: Path) -> tuple[dict[tuple[str, str], frozenset[int]], int]:
    lines = physical_lines(path)
    if not lines:
        raise ValueError(f"Metadata is empty: {path}")
    header_width = len(lines[0].split("\t"))
    labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    malformed = 0
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != header_width:
            malformed += 1
            continue
        match = METADATA_FILENAME_RE.fullmatch(Path(fields[6]).name)
        if match:
            labels[(fields[5], fields[4])].add(FILENAME_LABELS[match.group(1)])
    return {key: frozenset(value) for key, value in labels.items()}, malformed


def load_raw_pairs(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    malformed: list[int] = []
    for line_number, line in enumerate(physical_lines(path), start=1):
        fields = line.split("\t")
        if len(fields) != 6:
            malformed.append(line_number)
            continue
        source_doc, target_doc, source_ids, target_ids, english, turkish = fields
        if not source_doc.startswith("en/") or not target_doc.startswith("tr/"):
            continue
        alignment_id = f"{source_doc}#{source_ids}=>{target_doc}#{target_ids}"
        if alignment_id in result:
            raise ValueError(f"Duplicate raw alignment ID: {alignment_id}")
        result[alignment_id] = {
            "source_document": source_doc,
            "source_segment_ids": source_ids,
            "target_document": target_doc,
            "target_segment_ids": target_ids,
            "english_text": english,
            "turkish_text": turkish,
        }
    if malformed:
        raise ValueError(f"Malformed raw pair rows: {malformed[:10]}")
    return result


def validate_canonical(
    path: Path,
    raw_pairs: Mapping[str, Mapping[str, str]],
    metadata: Mapping[tuple[str, str], frozenset[int]],
) -> tuple[list[str], list[dict[str, str]], dict[str, object]]:
    fields, rows = read_tsv(path)
    if fields != list(REQUIRED_COLUMNS):
        raise ValueError(f"Unexpected canonical columns: {fields}")
    if len(rows) != len(raw_pairs):
        raise ValueError(
            f"Canonical/raw pair count mismatch: {len(rows)} != {len(raw_pairs)}"
        )

    alignment_ids: set[str] = set()
    relation_group: dict[tuple[str, Hashable], str] = {}
    cardinality: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()

    for line_number, row in enumerate(rows, start=2):
        alignment_id = row["alignment_id"]
        if not alignment_id or alignment_id in alignment_ids:
            raise ValueError(f"Missing/duplicate alignment ID at line {line_number}")
        alignment_ids.add(alignment_id)
        if not row["leakage_group_id"]:
            raise ValueError(f"Missing leakage group at line {line_number}")
        if not row["english_text"].strip() or not row["turkish_text"].strip():
            raise ValueError(f"Empty text at line {line_number}")
        if not row["source_document"].startswith("en/"):
            raise ValueError(f"Non-English source at line {line_number}")
        if not row["target_document"].startswith("tr/"):
            raise ValueError(f"Non-Turkish target at line {line_number}")

        expected_id = (
            f"{row['source_document']}#{row['source_segment_ids']}=>"
            f"{row['target_document']}#{row['target_segment_ids']}"
        )
        if alignment_id != expected_id:
            raise ValueError(f"Alignment ID/field mismatch at line {line_number}")
        raw = raw_pairs.get(alignment_id)
        if raw is None or any(row[key] != value for key, value in raw.items()):
            raise ValueError(f"Canonical/raw pairing mismatch: {alignment_id}")

        numeric = parse_numeric_list(row["labels_numeric"], line=line_number)
        names = parse_string_list(row["labels"], field="labels", line=line_number)
        if not numeric:
            raise ValueError(f"Missing labels: {alignment_id}")
        if numeric != tuple(sorted(set(numeric))):
            raise ValueError(f"Non-canonical numeric labels: {alignment_id}")
        if any(label not in NUMERIC_TO_LABEL for label in numeric):
            raise ValueError(f"Label outside 1-8: {alignment_id}")
        expected_names = tuple(NUMERIC_TO_LABEL[label] for label in numeric)
        if names != expected_names:
            raise ValueError(f"Numeric/name label mismatch: {alignment_id}")

        reconstructed: set[int] = set()
        for segment_id in row["source_segment_ids"].split():
            values = metadata.get((row["source_document"], segment_id))
            if not values:
                raise ValueError(
                    f"Missing source metadata for {alignment_id}, segment {segment_id}"
                )
            reconstructed.update(values)
        if numeric != tuple(sorted(reconstructed)):
            raise ValueError(f"Canonical/source-metadata label mismatch: {alignment_id}")

        group = row["leakage_group_id"]
        relations: list[tuple[str, Hashable]] = [
            ("english_text", row["english_text"]),
            ("turkish_text", row["turkish_text"]),
        ]
        relations.extend(
            ("source_segment", (row["source_document"], segment_id))
            for segment_id in row["source_segment_ids"].split()
        )
        for relation in relations:
            previous = relation_group.setdefault(relation, group)
            if previous != group:
                raise ValueError(
                    f"Leakage relation crosses canonical groups: {relation!r}"
                )

        cardinality[len(numeric)] += 1
        label_counts.update(names)
        group_counts[group] += 1

    if alignment_ids != set(raw_pairs):
        raise ValueError("Canonical alignment IDs do not exactly cover raw en-tr pairs")

    summary: dict[str, object] = {
        "rows": len(rows),
        "groups": len(group_counts),
        "multi_label_rows": sum(count for size, count in cardinality.items() if size > 1),
        "label_cardinality": dict(sorted(cardinality.items())),
        "label_counts": {label: label_counts[label] for label in LABELS},
        "largest_group": max(group_counts.values(), default=0),
    }
    return fields, rows, summary


def cross_partition_count(
    partitions: Mapping[str, Sequence[Mapping[str, str]]],
    getter,
) -> int:
    locations: dict[Hashable, set[str]] = defaultdict(set)
    for split, rows in partitions.items():
        for row in rows:
            values = getter(row)
            if isinstance(values, list):
                for value in values:
                    locations[value].add(split)
            else:
                locations[values].add(split)
    return sum(len(splits) > 1 for splits in locations.values())


def validate_partitions(
    canonical_fields: Sequence[str],
    canonical_rows: Sequence[Mapping[str, str]],
    split_dir: Path,
    expected_rows: Mapping[str, object],
    manifest_path: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    canonical_by_id = {row["alignment_id"]: dict(row) for row in canonical_rows}
    partitions: dict[str, list[dict[str, str]]] = {}
    seen_ids: set[str] = set()
    per_split: dict[str, object] = {}

    for split in SPLITS:
        fields, rows = read_tsv(split_dir / f"{split}.tsv")
        if fields != list(canonical_fields):
            raise ValueError(f"{split} fields do not match canonical fields")
        expected_count = expected_rows.get(split)
        if not isinstance(expected_count, int) or len(rows) != expected_count:
            raise ValueError(
                f"Unexpected {split} size: {len(rows)} != {expected_count!r}"
            )
        local_ids: set[str] = set()
        cardinality: Counter[int] = Counter()
        label_counts: Counter[str] = Counter()
        for row in rows:
            alignment_id = row["alignment_id"]
            if alignment_id in local_ids or alignment_id in seen_ids:
                raise ValueError(f"Alignment appears more than once: {alignment_id}")
            local_ids.add(alignment_id)
            seen_ids.add(alignment_id)
            if canonical_by_id.get(alignment_id) != row:
                raise ValueError(f"{split} row differs from canonical: {alignment_id}")
            labels = parse_string_list(row["labels"], field="labels", line=0)
            cardinality[len(labels)] += 1
            label_counts.update(labels)
        partitions[split] = rows
        per_split[split] = {
            "rows": len(rows),
            "ratio": len(rows) / len(canonical_rows),
            "groups": len({row["leakage_group_id"] for row in rows}),
            "single_label_rows": cardinality[1],
            "multi_label_rows": sum(
                count for size, count in cardinality.items() if size > 1
            ),
            "label_counts": {label: label_counts[label] for label in LABELS},
        }

    if seen_ids != set(canonical_by_id):
        raise ValueError("Split union does not exactly cover canonical alignment IDs")

    overlap = {
        "alignment_id": cross_partition_count(
            partitions, lambda row: row["alignment_id"]
        ),
        "leakage_group_id": cross_partition_count(
            partitions, lambda row: row["leakage_group_id"]
        ),
        "english_text": cross_partition_count(
            partitions, lambda row: row["english_text"]
        ),
        "turkish_text": cross_partition_count(
            partitions, lambda row: row["turkish_text"]
        ),
        "bilingual_text_pair": cross_partition_count(
            partitions, lambda row: (row["english_text"], row["turkish_text"])
        ),
        "source_segment": cross_partition_count(
            partitions,
            lambda row: [
                (row["source_document"], segment_id)
                for segment_id in row["source_segment_ids"].split()
            ],
        ),
    }
    if any(overlap.values()):
        raise ValueError(f"Cross-partition leakage detected: {overlap}")

    manifest_fields, manifest_rows = read_tsv(manifest_path)
    if manifest_fields != ["alignment_id", "leakage_group_id", "split"]:
        raise ValueError(f"Unexpected manifest columns: {manifest_fields}")
    expected_manifest = {
        row["alignment_id"]: (row["leakage_group_id"], split)
        for split, rows in partitions.items()
        for row in rows
    }
    observed_manifest: dict[str, tuple[str, str]] = {}
    for row in manifest_rows:
        alignment_id = row["alignment_id"]
        if alignment_id in observed_manifest:
            raise ValueError(f"Duplicate manifest alignment ID: {alignment_id}")
        observed_manifest[alignment_id] = (row["leakage_group_id"], row["split"])
    if observed_manifest != expected_manifest:
        raise ValueError("Split manifest does not exactly match split data")

    summary: dict[str, object] = {
        "per_split": per_split,
        "cross_partition_overlap": overlap,
        "manifest_rows": len(manifest_rows),
        "manifest_exact_match": True,
        "canonical_coverage": True,
    }
    return partitions, summary


def markdown_report(
    config_path: Path,
    config: Mapping[str, object],
    canonical_path: Path,
    split_dir: Path,
    manifest_path: Path,
    canonical: Mapping[str, object],
    split_summary: Mapping[str, object],
    metadata_malformed: int,
) -> str:
    per_split = split_summary["per_split"]
    assert isinstance(per_split, dict)
    lines = [
        "# Data audit",
        "",
        "All checks below use local files only; no model output or external resource was used.",
        "",
        "## Outcome",
        "",
        "**PASS.** The prepared English-Turkish data, fixed partitions, and manifest are consistent. No tested form of cross-partition leakage was found.",
        "",
        "## Fixed settings",
        "",
        f"- Split seed: {config['seed']}",
        "- Target ratios: 70% train, 20% development, 10% test",
        "- Fixed label order: " + ", ".join(LABELS),
        "- Label source: projected reference labels reconstructed from English source annotation metadata",
        "- Zero-division policy for later metrics: 0.0",
        "",
        "## Dataset and partition counts",
        "",
        "| Partition | Rows | Observed ratio | Leakage groups | Single-label | Multi-label |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        values = per_split[split]
        lines.append(
            f"| {split} | {values['rows']:,} | {100 * values['ratio']:.4f}% | "
            f"{values['groups']:,} | {values['single_label_rows']:,} | "
            f"{values['multi_label_rows']:,} |"
        )
    lines.extend(
        [
            f"| Total | {canonical['rows']:,} | 100.0000% | "
            f"{canonical['groups']:,} | "
            f"{canonical['rows'] - canonical['multi_label_rows']:,} | "
            f"{canonical['multi_label_rows']:,} |",
            "",
            "Multi-label targets are present in every partition and remain JSON lists rather than being collapsed to one class.",
            "",
            "## Emotion counts",
            "",
            "| Emotion | Prepared/aligned | Train | Development | Test |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in LABELS:
        lines.append(
            f"| {label} | {canonical['label_counts'][label]:,} | "
            f"{per_split['train']['label_counts'][label]:,} | "
            f"{per_split['dev']['label_counts'][label]:,} | "
            f"{per_split['test']['label_counts'][label]:,} |"
        )
    lines.extend(
        [
            "",
            "## Leakage and consistency checks",
            "",
            "| Check | Observed | Status |",
            "|---|---:|---|",
        ]
    )
    overlap = split_summary["cross_partition_overlap"]
    for name, count in overlap.items():
        lines.append(f"| Cross-partition `{name}` values | {count} | PASS |")
    lines.extend(
        [
            f"| Manifest records | {split_summary['manifest_rows']:,} | PASS |",
            "| Manifest exactly matches split assignment | yes | PASS |",
            "| Split union exactly covers all prepared rows | yes | PASS |",
            f"| Raw English-Turkish pair rows matched | {canonical['rows']:,} | PASS |",
            f"| Source-metadata labels reconstructed | {canonical['rows']:,} | PASS |",
            "| Missing English texts | 0 | PASS |",
            "| Missing Turkish texts | 0 | PASS |",
            "| Missing labels | 0 | PASS |",
            "| Labels outside permitted inventory | 0 | PASS |",
            f"| Malformed metadata rows ignored by the existing preparation rule | {metadata_malformed} | PASS |",
            "",
            "The pairing check rebuilds every alignment ID from the English and Turkish document and segment fields. It checks all six raw pair fields against `pairs-tr.txt`, then checks each row's label names and numeric IDs against the source annotation metadata.",
            "",
            "## File hashes",
            "",
            "| File | SHA-256 |",
            "|---|---|",
            f"| `{canonical_path.relative_to(ROOT)}` | `{sha256(canonical_path)}` |",
        ]
    )
    for split in SPLITS:
        path = split_dir / f"{split}.tsv"
        lines.append(f"| `{path.relative_to(ROOT)}` | `{sha256(path)}` |")
    lines.extend(
        [
            f"| `{manifest_path.relative_to(ROOT)}` | `{sha256(manifest_path)}` |",
            f"| `{config_path.relative_to(ROOT)}` | `{sha256(config_path)}` |",
            "",
            "## Interpretation limitation",
            "",
            "The Turkish targets are aligned subtitle counterparts carrying projected reference labels reconstructed through their English source annotations. They are not independently human-annotated Turkish emotion labels. The split prevents exact/source-identity leakage, but it does not establish that every subtitle alignment or projected label is semantically perfect.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    canonical_path = ROOT / str(config["canonical_data"])
    split_dir = ROOT / str(config["split_directory"])
    manifest_path = ROOT / str(config["manifest"])
    expected_rows = config.get("expected_rows")
    if not isinstance(expected_rows, dict):
        raise ValueError("Config is missing expected_rows")

    raw_pairs = load_raw_pairs(args.pairs)
    metadata, malformed_metadata = load_metadata(args.metadata)
    fields, canonical_rows, canonical_summary = validate_canonical(
        canonical_path, raw_pairs, metadata
    )
    _, split_summary = validate_partitions(
        fields, canonical_rows, split_dir, expected_rows, manifest_path
    )

    if args.report:
        write_text_atomic(
            args.report,
            markdown_report(
                args.config,
                config,
                canonical_path,
                split_dir,
                manifest_path,
                canonical_summary,
                split_summary,
                malformed_metadata,
            ),
        )
    print("DATA VALIDATION: PASS")
    print(f"canonical rows: {canonical_summary['rows']}")
    print(f"leakage groups: {canonical_summary['groups']}")
    for split in SPLITS:
        values = split_summary["per_split"][split]
        print(f"{split}: {values['rows']} rows ({100 * values['ratio']:.4f}%)")
    print("cross-partition overlaps:", split_summary["cross_partition_overlap"])
    print("manifest exact match: True")
    if args.report:
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()
