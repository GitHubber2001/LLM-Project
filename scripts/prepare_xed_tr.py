#!/usr/bin/env python3
"""Build the canonical English-Turkish XED alignment dataset.

Methodology:
- Read the untouched XED pair and annotation-metadata files.
- Retain only English-source to Turkish-target alignments.
- Reconstruct labels by English source document + segment ID.
- Union multiple emotion annotations into a canonical multi-label set.
- Keep every alignment row, including repeated sentence texts.
- Assign transitive leakage groups using source segment identity, exact English
  text, or exact Turkish text. Movie identity is deliberately not used.

No fuzzy matching, subtitle repair, deduplication, or dataset splitting occurs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "subtitle-retrieval" / "students" / "pairs-tr.txt"
DEFAULT_METADATA = (
    ROOT / "subtitle-retrieval" / "students" / "student_annotations.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "xed_en_tr_aligned.tsv"

LABEL_NAMES = {
    1: "anger",
    2: "anticipation",
    3: "disgust",
    4: "fear",
    5: "joy",
    6: "sadness",
    7: "surprise",
    8: "trust",
}
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
OUTPUT_COLUMNS = [
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
]


@dataclass(frozen=True)
class PairRow:
    input_line_number: int
    source_document: str
    target_document: str
    source_segment_ids: str
    target_segment_ids: str
    english_text: str
    turkish_text: str

    @property
    def source_language(self) -> str:
        return self.source_document.split("/", 1)[0]

    @property
    def target_language(self) -> str:
        return self.target_document.split("/", 1)[0]

    @property
    def alignment_id(self) -> str:
        return (
            f"{self.source_document}#{self.source_segment_ids}=>"
            f"{self.target_document}#{self.target_segment_ids}"
        )

    @property
    def source_segment_keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (self.source_document, segment_id)
            for segment_id in self.source_segment_ids.split()
        )


@dataclass(frozen=True)
class PreparedRow:
    pair: PairRow
    labels_numeric: tuple[int, ...]
    labels: tuple[str, ...]
    leakage_group_id: str

    def as_tsv_row(self) -> list[str]:
        return [
            self.pair.alignment_id,
            self.leakage_group_id,
            self.pair.source_document,
            self.pair.source_segment_ids,
            self.pair.target_document,
            self.pair.target_segment_ids,
            self.pair.english_text,
            self.pair.turkish_text,
            json.dumps(self.labels_numeric),
            json.dumps(self.labels),
        ]


@dataclass(frozen=True)
class MetadataResult:
    labels_by_source_segment: dict[tuple[str, str], frozenset[int]]
    physical_data_rows: int
    well_formed_rows: int
    malformed_rows: tuple[tuple[int, int], ...]
    unrecognized_filename_rows: tuple[int, ...]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.component_size = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.component_size[left_root] < self.component_size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] += self.component_size[right_root]


def physical_lines(path: Path) -> list[str]:
    """Split only at LF; quotes remain literal and a final row needs no LF."""
    records = path.read_bytes().split(b"\n")
    if records and records[-1] == b"":
        records.pop()
    return [record.removesuffix(b"\r").decode("utf-8") for record in records]


def read_pairs(path: Path) -> tuple[list[PairRow], Counter[str]]:
    rows: list[PairRow] = []
    source_languages: Counter[str] = Counter()
    malformed: list[tuple[int, int]] = []

    for line_number, line in enumerate(physical_lines(path), start=1):
        fields = line.split("\t")
        if len(fields) != 6:
            malformed.append((line_number, len(fields)))
            continue
        row = PairRow(line_number, *fields)
        source_languages[row.source_language] += 1
        if row.source_language == "en" and row.target_language == "tr":
            rows.append(row)

    if malformed:
        preview = ", ".join(f"line {line}: {width} fields" for line, width in malformed[:10])
        raise ValueError(f"Malformed pairs TSV records ({len(malformed)}): {preview}")
    return rows, source_languages


def read_metadata(path: Path) -> MetadataResult:
    lines = physical_lines(path)
    if not lines:
        raise ValueError(f"Metadata file is empty: {path}")
    header = lines[0].split("\t")
    if len(header) < 7:
        raise ValueError(f"Unexpected metadata header: {header!r}")

    labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    malformed: list[tuple[int, int]] = []
    unrecognized: list[int] = []
    well_formed = 0

    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != len(header):
            malformed.append((line_number, len(fields)))
            continue
        well_formed += 1
        source_document = fields[5]
        source_segment_id = fields[4]
        match = METADATA_FILENAME_RE.fullmatch(Path(fields[6]).name)
        if not match:
            unrecognized.append(line_number)
            continue
        label = FILENAME_LABELS[match.group(1)]
        labels[(source_document, source_segment_id)].add(label)

    return MetadataResult(
        labels_by_source_segment={
            key: frozenset(value) for key, value in labels.items()
        },
        physical_data_rows=len(lines) - 1,
        well_formed_rows=well_formed,
        malformed_rows=tuple(malformed),
        unrecognized_filename_rows=tuple(unrecognized),
    )


def reconstruct_labels(
    pair: PairRow,
    labels_by_source_segment: dict[tuple[str, str], frozenset[int]],
) -> tuple[int, ...]:
    labels: set[int] = set()
    missing_keys: list[tuple[str, str]] = []
    for key in pair.source_segment_keys:
        segment_labels = labels_by_source_segment.get(key)
        if not segment_labels:
            missing_keys.append(key)
        else:
            labels.update(segment_labels)
    if missing_keys:
        raise ValueError(
            f"No reconstructed labels for alignment {pair.alignment_id}; "
            f"missing metadata keys: {missing_keys}"
        )
    return tuple(sorted(labels))


def assign_leakage_groups(pairs: list[PairRow]) -> list[str]:
    union_find = UnionFind(len(pairs))
    seen_source_segment: dict[tuple[str, str], int] = {}
    seen_english: dict[str, int] = {}
    seen_turkish: dict[str, int] = {}

    def connect(index: int, seen: dict[Hashable, int], value: Hashable) -> None:
        previous = seen.get(value)
        if previous is None:
            seen[value] = index
        else:
            union_find.union(index, previous)

    for index, pair in enumerate(pairs):
        for source_segment_key in pair.source_segment_keys:
            connect(index, seen_source_segment, source_segment_key)
        connect(index, seen_english, pair.english_text)
        connect(index, seen_turkish, pair.turkish_text)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(pairs)):
        members_by_root[union_find.find(index)].append(index)

    # Stable under repeat runs on the same raw file: components are numbered by
    # the earliest retained alignment's original order.
    ordered_components = sorted(members_by_root.values(), key=min)
    width = max(6, len(str(len(ordered_components))))
    group_ids = [""] * len(pairs)
    for number, member_indices in enumerate(ordered_components, start=1):
        group_id = f"leakage_group_{number:0{width}d}"
        for index in member_indices:
            group_ids[index] = group_id
    return group_ids


def validate_prepared_rows(rows: list[PreparedRow]) -> None:
    alignment_ids = [row.pair.alignment_id for row in rows]
    duplicates = [
        alignment_id
        for alignment_id, count in Counter(alignment_ids).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(
            f"Duplicate alignment IDs detected ({len(duplicates)}): {duplicates[:10]}"
        )

    for row in rows:
        if not row.pair.english_text.strip():
            raise ValueError(f"Missing English text: {row.pair.alignment_id}")
        if not row.pair.turkish_text.strip():
            raise ValueError(f"Missing Turkish text: {row.pair.alignment_id}")
        if not row.labels_numeric:
            raise ValueError(f"Missing labels: {row.pair.alignment_id}")
        outside = set(row.labels_numeric) - set(LABEL_NAMES)
        if outside:
            raise ValueError(
                f"Labels outside 1-8 for {row.pair.alignment_id}: {sorted(outside)}"
            )
        expected_names = tuple(LABEL_NAMES[label] for label in row.labels_numeric)
        if row.labels != expected_names:
            raise ValueError(
                f"Label-name mismatch for {row.pair.alignment_id}: "
                f"{row.labels!r} != {expected_names!r}"
            )
        if not row.leakage_group_id:
            raise ValueError(f"Missing leakage group: {row.pair.alignment_id}")

    # Verify that every direct leakage relation has exactly one group. Because
    # the assigned groups are union-find components, this also verifies the
    # transitive closure represented in the output.
    relation_groups: dict[tuple[str, Hashable], set[str]] = defaultdict(set)
    for row in rows:
        for key in row.pair.source_segment_keys:
            relation_groups[("source_segment", key)].add(row.leakage_group_id)
        relation_groups[("english_text", row.pair.english_text)].add(
            row.leakage_group_id
        )
        relation_groups[("turkish_text", row.pair.turkish_text)].add(
            row.leakage_group_id
        )
    broken = [key for key, groups in relation_groups.items() if len(groups) != 1]
    if broken:
        raise ValueError(
            f"Leakage grouping validation failed for {len(broken)} keys: {broken[:10]}"
        )


def write_tsv_atomic(path: Path, rows: list[PreparedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(
                handle,
                delimiter="\t",
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writerow(OUTPUT_COLUMNS)
            writer.writerows(row.as_tsv_row() for row in rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_written_tsv(path: Path, expected_rows: list[PreparedRow]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise ValueError(
                f"Output columns differ: {reader.fieldnames!r} != {OUTPUT_COLUMNS!r}"
            )
        loaded = list(reader)
    if len(loaded) != len(expected_rows):
        raise ValueError(
            f"Output row-count mismatch: {len(loaded)} != {len(expected_rows)}"
        )
    if len({row["alignment_id"] for row in loaded}) != len(loaded):
        raise ValueError("Duplicate alignment IDs detected after TSV round trip")

    for output_line, (loaded_row, expected) in enumerate(
        zip(loaded, expected_rows), start=2
    ):
        expected_row = dict(
            zip(OUTPUT_COLUMNS, expected.as_tsv_row(), strict=True)
        )
        if loaded_row != expected_row:
            differing_columns = [
                column
                for column in OUTPUT_COLUMNS
                if loaded_row.get(column) != expected_row[column]
            ]
            raise ValueError(
                f"Full-row TSV round-trip mismatch at output line {output_line}; "
                f"columns={differing_columns}, "
                f"expected_alignment_id={expected.pair.alignment_id!r}"
            )


def duplicate_count(values: Iterable[Hashable]) -> tuple[int, int, int]:
    """Repeated distinct values, rows involved, and excess occurrences."""
    repeated = [count for count in Counter(values).values() if count > 1]
    return len(repeated), sum(repeated), sum(count - 1 for count in repeated)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_summary(
    pairs_path: Path,
    metadata_path: Path,
    output_path: Path,
    rows: list[PreparedRow],
    source_languages: Counter[str],
    metadata: MetadataResult,
) -> None:
    group_sizes = Counter(row.leakage_group_id for row in rows)
    size_distribution = Counter(group_sizes.values())
    multi_group_distribution = {
        size: count for size, count in sorted(size_distribution.items()) if size > 1
    }
    label_cardinality = Counter(len(row.labels_numeric) for row in rows)
    label_frequency: Counter[int] = Counter()
    for row in rows:
        label_frequency.update(row.labels_numeric)

    missing_english = sum(not row.pair.english_text.strip() for row in rows)
    missing_turkish = sum(not row.pair.turkish_text.strip() for row in rows)
    missing_labels = sum(not row.labels_numeric for row in rows)
    outside_labels = sorted(
        {
            label
            for row in rows
            for label in row.labels_numeric
            if label not in LABEL_NAMES
        }
    )
    print("XED ENGLISH-TURKISH CANONICAL DATASET")
    print(f"output: {output_path}")
    print(f"Python version: {platform.python_version()}")
    print("SHA-256")
    print(f"  pairs-tr.txt: {sha256(pairs_path)}")
    print(f"  student_annotations.csv: {sha256(metadata_path)}")
    print(f"  {output_path.name}: {sha256(output_path)}")
    print()
    print("INPUT/FILTER")
    print(f"  original pairs source-language distribution: {dict(sorted(source_languages.items()))}")
    print(f"  retained filter: source=en and target=tr")
    print(f"  metadata physical data rows: {metadata.physical_data_rows}")
    print(f"  metadata well-formed rows: {metadata.well_formed_rows}")
    print(f"  metadata malformed physical rows ignored: {len(metadata.malformed_rows)}")
    print(f"  metadata rows with unrecognized filename category ignored: {len(metadata.unrecognized_filename_rows)}")
    print()
    print("OUTPUT VALIDATION")
    print(f"  total output rows: {len(rows)}")
    print(f"  unique alignment IDs: {len({row.pair.alignment_id for row in rows})}")
    print(f"  unique leakage groups: {len(group_sizes)}")
    print(f"  multi-row leakage groups: {sum(size > 1 for size in group_sizes.values())}")
    print(f"  rows in multi-row leakage groups: {sum(size for size in group_sizes.values() if size > 1)}")
    print(f"  multi-row leakage-group size distribution (size -> groups): {multi_group_distribution}")
    print(f"  largest leakage group: {max(group_sizes.values(), default=0)}")
    print(f"  single-label rows: {label_cardinality[1]}")
    print(f"  multi-label rows: {len(rows) - label_cardinality[1]}")
    print(f"  label-cardinality distribution: {dict(sorted(label_cardinality.items()))}")
    print("  emotion frequencies:")
    for label, name in LABEL_NAMES.items():
        print(f"    {label} {name}: {label_frequency[label]}")
    print(f"  missing English texts: {missing_english}")
    print(f"  missing Turkish texts: {missing_turkish}")
    print(f"  missing labels: {missing_labels}")
    print(f"  labels outside permitted 1-8: {outside_labels}")
    print("  retained exact-text repetitions (distinct repeated values, rows, excess):")
    print(
        "    English: "
        f"{duplicate_count(row.pair.english_text for row in rows)}"
    )
    print(
        "    Turkish: "
        f"{duplicate_count(row.pair.turkish_text for row in rows)}"
    )
    print(
        "    bilingual pairs: "
        f"{duplicate_count((row.pair.english_text, row.pair.turkish_text) for row in rows)}"
    )
    print()
    print("FIRST 10 GENERATED ROWS")
    print("\t".join(OUTPUT_COLUMNS))
    for row in rows[:10]:
        print("\t".join(row.as_tsv_row()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs, source_languages = read_pairs(args.pairs)
    metadata = read_metadata(args.metadata)
    leakage_group_ids = assign_leakage_groups(pairs)

    prepared: list[PreparedRow] = []
    for pair, leakage_group_id in zip(pairs, leakage_group_ids):
        labels_numeric = reconstruct_labels(
            pair, metadata.labels_by_source_segment
        )
        prepared.append(
            PreparedRow(
                pair=pair,
                labels_numeric=labels_numeric,
                labels=tuple(LABEL_NAMES[label] for label in labels_numeric),
                leakage_group_id=leakage_group_id,
            )
        )

    validate_prepared_rows(prepared)
    write_tsv_atomic(args.output, prepared)
    validate_written_tsv(args.output, prepared)
    print_summary(
        args.pairs,
        args.metadata,
        args.output,
        prepared,
        source_languages,
        metadata,
    )


if __name__ == "__main__":
    main()
