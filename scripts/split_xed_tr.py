#!/usr/bin/env python3
"""Create the fixed group-aware multi-label XED English-Turkish split.

Algorithm
---------
1. Treat every ``leakage_group_id`` as indivisible and calculate its row count
   and row-level positive count for each of the eight emotion labels.
2. Use a seeded, capacity-constrained greedy assignment. Candidate splits may
   not exceed their exact row targets. The greedy cost measures the weighted
   squared difference between the candidate split's partial label counts and
   the overall label prevalence at its partial row count.
3. Improve the completed assignment with seeded hill climbing. Each proposal
   swaps two groups of equal size in different splits, so exact row counts and
   group integrity are preserved. A swap is accepted only when it reduces the
   final weighted squared label-count error relative to proportional targets.

The procedure depends only on labels, leakage groups, and ``SEED = 42``. It
does not inspect model outputs or use movie identity. The canonical input is
read-only and all derived files are written atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "xed_en_tr_aligned.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "splits"

SEED = 42
SPLITS = ("train", "dev", "test")
TARGET_ROWS = {"train": 4852, "dev": 1386, "test": 693}
PERMITTED_LABELS = tuple(range(1, 9))
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
LOCAL_IMPROVEMENT_PROPOSALS = 500_000
REQUIRED_COLUMNS = {
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
}


@dataclass(frozen=True)
class CanonicalRow:
    values: dict[str, str]
    labels: tuple[int, ...]
    label_names: tuple[str, ...]

    @property
    def alignment_id(self) -> str:
        return self.values["alignment_id"]

    @property
    def leakage_group_id(self) -> str:
        return self.values["leakage_group_id"]

    @property
    def english_text(self) -> str:
        return self.values["english_text"]

    @property
    def turkish_text(self) -> str:
        return self.values["turkish_text"]

    @property
    def source_segment_keys(self) -> tuple[tuple[str, str], ...]:
        document = self.values["source_document"]
        return tuple(
            (document, segment_id)
            for segment_id in self.values["source_segment_ids"].split()
        )


@dataclass(frozen=True)
class LeakageGroup:
    group_id: str
    row_indices: tuple[int, ...]
    row_count: int
    label_counts: tuple[int, ...]


@dataclass(frozen=True)
class AssignmentResult:
    group_to_split: dict[str, str]
    initial_objective: float
    final_objective: float
    accepted_swaps: int


@dataclass(frozen=True)
class ValidationResult:
    alignment_overlap: int
    leakage_group_overlap: int
    english_text_overlap: int
    turkish_text_overlap: int
    source_key_overlap: int


def read_canonical(path: Path) -> tuple[list[CanonicalRow], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Canonical dataset has no header: {path}")
        missing_columns = REQUIRED_COLUMNS - set(fieldnames)
        if missing_columns:
            raise ValueError(
                f"Canonical dataset is missing columns: {sorted(missing_columns)}"
            )

        rows: list[CanonicalRow] = []
        for file_line, values in enumerate(reader, start=2):
            try:
                raw_labels = json.loads(values["labels_numeric"])
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(
                    f"Invalid labels_numeric JSON at line {file_line}: "
                    f"{values.get('labels_numeric')!r}"
                ) from error
            if not isinstance(raw_labels, list) or not all(
                isinstance(label, int) and not isinstance(label, bool)
                for label in raw_labels
            ):
                raise ValueError(
                    f"labels_numeric must be an integer JSON list at line {file_line}"
                )
            try:
                raw_label_names = json.loads(values["labels"])
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(
                    f"Invalid labels JSON at line {file_line}: "
                    f"{values.get('labels')!r}"
                ) from error
            if not isinstance(raw_label_names, list) or not all(
                isinstance(label_name, str) for label_name in raw_label_names
            ):
                raise ValueError(
                    f"labels must be a string JSON list at line {file_line}"
                )
            labels = tuple(raw_labels)
            rows.append(
                CanonicalRow(dict(values), labels, tuple(raw_label_names))
            )
    return rows, list(fieldnames)


def validate_canonical(rows: Sequence[CanonicalRow]) -> None:
    expected_total = sum(TARGET_ROWS.values())
    if len(rows) != expected_total:
        raise ValueError(
            f"Expected {expected_total} canonical rows, found {len(rows)}"
        )

    alignment_counts = Counter(row.alignment_id for row in rows)
    duplicates = [key for key, count in alignment_counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Canonical alignment IDs are not unique ({len(duplicates)}): "
            f"{duplicates[:10]}"
        )

    for row in rows:
        if not row.alignment_id:
            raise ValueError("Canonical row has an empty alignment_id")
        if not row.leakage_group_id:
            raise ValueError(f"Missing leakage group: {row.alignment_id}")
        if not row.english_text.strip():
            raise ValueError(f"Missing English text: {row.alignment_id}")
        if not row.turkish_text.strip():
            raise ValueError(f"Missing Turkish text: {row.alignment_id}")
        if not row.source_segment_keys:
            raise ValueError(f"Missing English source segment ID: {row.alignment_id}")
        if not row.labels:
            raise ValueError(f"Missing labels: {row.alignment_id}")
        if tuple(sorted(set(row.labels))) != row.labels:
            raise ValueError(
                f"Labels are not a sorted unique set: {row.alignment_id} {row.labels}"
            )
        outside = set(row.labels) - set(PERMITTED_LABELS)
        if outside:
            raise ValueError(
                f"Labels outside 1-8: {row.alignment_id} {sorted(outside)}"
            )
        expected_names = tuple(LABEL_NAMES[label] for label in row.labels)
        if row.label_names != expected_names:
            raise ValueError(
                f"labels do not match labels_numeric in canonical numeric order: "
                f"{row.alignment_id}; {row.label_names!r} != {expected_names!r}"
            )


def build_groups(rows: Sequence[CanonicalRow]) -> list[LeakageGroup]:
    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_group[row.leakage_group_id].append(index)

    groups: list[LeakageGroup] = []
    for group_id, indices in indices_by_group.items():
        label_counts = [0] * len(PERMITTED_LABELS)
        for index in indices:
            for label in rows[index].labels:
                label_counts[label - 1] += 1
        groups.append(
            LeakageGroup(
                group_id=group_id,
                row_indices=tuple(indices),
                row_count=len(indices),
                label_counts=tuple(label_counts),
            )
        )
    return groups


def overall_label_counts(groups: Iterable[LeakageGroup]) -> tuple[int, ...]:
    counts = [0] * len(PERMITTED_LABELS)
    for group in groups:
        for index, count in enumerate(group.label_counts):
            counts[index] += count
    return tuple(counts)


def final_objective(
    split_label_counts: Sequence[Sequence[int]],
    overall_counts: Sequence[int],
    total_rows: int,
) -> float:
    """Weighted squared label-count error against proportional targets."""
    objective = 0.0
    for split_index, split in enumerate(SPLITS):
        target_rows = TARGET_ROWS[split]
        for label_index, overall_count in enumerate(overall_counts):
            expected = overall_count * target_rows / total_rows
            difference = split_label_counts[split_index][label_index] - expected
            objective += difference * difference / max(expected, 1.0)
    return objective


def assign_groups(
    groups: Sequence[LeakageGroup], total_rows: int
) -> AssignmentResult:
    rng = random.Random(SEED)
    overall_counts = overall_label_counts(groups)
    prevalence = [count / total_rows for count in overall_counts]
    split_rows = [0] * len(SPLITS)
    split_label_counts = [
        [0] * len(PERMITTED_LABELS) for _ in range(len(SPLITS))
    ]
    split_index_by_group: dict[str, int] = {}

    # Fix iteration order before consuming seeded random numbers. Assignment
    # therefore depends on group IDs/features and the seed, not input row order.
    stable_groups = sorted(groups, key=lambda group: group.group_id)
    # Seeded jitter is used only for deterministic tie-breaking after group
    # size and inverse-frequency label information.
    jitter = {group.group_id: rng.random() for group in stable_groups}
    ordered_groups = sorted(
        stable_groups,
        key=lambda group: (
            -group.row_count,
            -sum(
                count / overall_counts[index]
                for index, count in enumerate(group.label_counts)
                if overall_counts[index]
            ),
            jitter[group.group_id],
            group.group_id,
        ),
    )

    def partial_composition_cost(split_index: int, group: LeakageGroup) -> float:
        new_rows = split_rows[split_index] + group.row_count
        cost = 0.0
        for label_index in range(len(PERMITTED_LABELS)):
            expected = prevalence[label_index] * new_rows
            actual = (
                split_label_counts[split_index][label_index]
                + group.label_counts[label_index]
            )
            difference = actual - expected
            cost += difference * difference / max(expected, 1.0)
        return cost

    for group in ordered_groups:
        candidates = [
            split_index
            for split_index, split in enumerate(SPLITS)
            if split_rows[split_index] + group.row_count <= TARGET_ROWS[split]
        ]
        if not candidates:
            raise RuntimeError(
                f"No split has capacity for indivisible group {group.group_id} "
                f"of size {group.row_count}; current rows={split_rows}"
            )
        candidate_jitter = [rng.random() for _ in SPLITS]
        chosen = min(
            candidates,
            key=lambda split_index: (
                partial_composition_cost(split_index, group),
                (split_rows[split_index] + group.row_count)
                / TARGET_ROWS[SPLITS[split_index]],
                candidate_jitter[split_index],
                split_index,
            ),
        )
        split_index_by_group[group.group_id] = chosen
        split_rows[chosen] += group.row_count
        for label_index, count in enumerate(group.label_counts):
            split_label_counts[chosen][label_index] += count

    if split_rows != [TARGET_ROWS[split] for split in SPLITS]:
        raise RuntimeError(
            f"Greedy assignment did not reach exact target rows: {split_rows}"
        )

    initial_objective = final_objective(
        split_label_counts, overall_counts, total_rows
    )

    groups_by_size: dict[int, list[LeakageGroup]] = defaultdict(list)
    for group in stable_groups:
        groups_by_size[group.row_count].append(group)
    swappable_sizes = [
        size
        for size, bucket in groups_by_size.items()
        if len({split_index_by_group[group.group_id] for group in bucket}) > 1
    ]
    if not swappable_sizes:
        raise RuntimeError("No equal-size cross-split group swaps are available")

    accepted_swaps = 0
    for _ in range(LOCAL_IMPROVEMENT_PROPOSALS):
        size = rng.choice(swappable_sizes)
        bucket = groups_by_size[size]
        left = rng.choice(bucket)
        right = rng.choice(bucket)
        left_split = split_index_by_group[left.group_id]
        right_split = split_index_by_group[right.group_id]
        if left_split == right_split:
            continue

        objective_delta = 0.0
        for label_index, overall_count in enumerate(overall_counts):
            left_expected = (
                overall_count * TARGET_ROWS[SPLITS[left_split]] / total_rows
            )
            right_expected = (
                overall_count * TARGET_ROWS[SPLITS[right_split]] / total_rows
            )
            left_old_difference = (
                split_label_counts[left_split][label_index] - left_expected
            )
            right_old_difference = (
                split_label_counts[right_split][label_index] - right_expected
            )
            left_new_difference = (
                left_old_difference
                - left.label_counts[label_index]
                + right.label_counts[label_index]
            )
            right_new_difference = (
                right_old_difference
                - right.label_counts[label_index]
                + left.label_counts[label_index]
            )
            objective_delta += (
                left_new_difference**2 - left_old_difference**2
            ) / max(left_expected, 1.0)
            objective_delta += (
                right_new_difference**2 - right_old_difference**2
            ) / max(right_expected, 1.0)

        if objective_delta < -1e-12:
            split_index_by_group[left.group_id] = right_split
            split_index_by_group[right.group_id] = left_split
            for label_index in range(len(PERMITTED_LABELS)):
                split_label_counts[left_split][label_index] += (
                    right.label_counts[label_index]
                    - left.label_counts[label_index]
                )
                split_label_counts[right_split][label_index] += (
                    left.label_counts[label_index]
                    - right.label_counts[label_index]
                )
            accepted_swaps += 1

    final_value = final_objective(
        split_label_counts, overall_counts, total_rows
    )
    if final_value > initial_objective + 1e-12:
        raise RuntimeError(
            f"Local improvement worsened objective: {initial_objective} -> {final_value}"
        )

    return AssignmentResult(
        group_to_split={
            group_id: SPLITS[split_index]
            for group_id, split_index in split_index_by_group.items()
        },
        initial_objective=initial_objective,
        final_objective=final_value,
        accepted_swaps=accepted_swaps,
    )


def rows_by_split(
    rows: Sequence[CanonicalRow], assignment: AssignmentResult
) -> dict[str, list[CanonicalRow]]:
    result = {split: [] for split in SPLITS}
    for row in rows:
        split = assignment.group_to_split.get(row.leakage_group_id)
        if split is None:
            raise ValueError(
                f"Leakage group has no split assignment: {row.leakage_group_id}"
            )
        result[split].append(row)
    return result


def cross_split_values(
    partitions: dict[str, list[CanonicalRow]],
    getter,
) -> int:
    splits_by_value: dict[Hashable, set[str]] = defaultdict(set)
    for split, rows in partitions.items():
        for row in rows:
            values = getter(row)
            if isinstance(values, tuple) and values and isinstance(values[0], tuple):
                for value in values:
                    splits_by_value[value].add(split)
            else:
                splits_by_value[values].add(split)
    return sum(len(splits) > 1 for splits in splits_by_value.values())


def validate_split(
    canonical_rows: Sequence[CanonicalRow],
    partitions: dict[str, list[CanonicalRow]],
) -> ValidationResult:
    canonical_ids = {row.alignment_id for row in canonical_rows}
    partition_ids = {
        split: {row.alignment_id for row in rows}
        for split, rows in partitions.items()
    }
    all_partition_ids = set().union(*partition_ids.values())
    if all_partition_ids != canonical_ids:
        missing = canonical_ids - all_partition_ids
        extra = all_partition_ids - canonical_ids
        raise ValueError(
            f"Canonical coverage failure: missing={len(missing)}, extra={len(extra)}"
        )
    if sum(len(rows) for rows in partitions.values()) != len(canonical_rows):
        raise ValueError("Canonical alignments do not appear exactly once")

    for split in SPLITS:
        if len(partitions[split]) != TARGET_ROWS[split]:
            raise ValueError(
                f"{split} row count {len(partitions[split])} != {TARGET_ROWS[split]}"
            )

    alignment_overlap = sum(
        len(partition_ids[SPLITS[left]] & partition_ids[SPLITS[right]])
        for left in range(len(SPLITS))
        for right in range(left + 1, len(SPLITS))
    )
    leakage_group_overlap = cross_split_values(
        partitions, lambda row: row.leakage_group_id
    )
    english_text_overlap = cross_split_values(
        partitions, lambda row: row.english_text
    )
    turkish_text_overlap = cross_split_values(
        partitions, lambda row: row.turkish_text
    )
    source_key_overlap = cross_split_values(
        partitions, lambda row: row.source_segment_keys
    )

    overlap_metrics = {
        "alignment": alignment_overlap,
        "leakage group": leakage_group_overlap,
        "English text": english_text_overlap,
        "Turkish text": turkish_text_overlap,
        "source key": source_key_overlap,
    }
    nonzero = {name: value for name, value in overlap_metrics.items() if value}
    if nonzero:
        raise ValueError(f"Cross-split leakage detected: {nonzero}")

    for split, rows in partitions.items():
        labels_in_split: set[int] = set()
        for row in rows:
            if not row.english_text.strip() or not row.turkish_text.strip():
                raise ValueError(f"Missing text in {split}: {row.alignment_id}")
            if not row.labels:
                raise ValueError(f"Missing label in {split}: {row.alignment_id}")
            outside = set(row.labels) - set(PERMITTED_LABELS)
            if outside:
                raise ValueError(
                    f"Invalid labels in {split}: {row.alignment_id} {outside}"
                )
            labels_in_split.update(row.labels)
        missing_labels = set(PERMITTED_LABELS) - labels_in_split
        if missing_labels:
            raise ValueError(
                f"Not all eight labels occur in {split}; missing {sorted(missing_labels)}"
            )

    return ValidationResult(
        alignment_overlap,
        leakage_group_overlap,
        english_text_overlap,
        turkish_text_overlap,
        source_key_overlap,
    )


def write_tsv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(
    output_dir: Path,
    fieldnames: Sequence[str],
    canonical_rows: Sequence[CanonicalRow],
    partitions: dict[str, list[CanonicalRow]],
    assignment: AssignmentResult,
) -> dict[str, Path]:
    paths = {
        "manifest": output_dir / "split_manifest.tsv",
        "train": output_dir / "train.tsv",
        "dev": output_dir / "dev.tsv",
        "test": output_dir / "test.tsv",
    }
    write_tsv_atomic(
        paths["manifest"],
        ["alignment_id", "leakage_group_id", "split"],
        (
            {
                "alignment_id": row.alignment_id,
                "leakage_group_id": row.leakage_group_id,
                "split": assignment.group_to_split[row.leakage_group_id],
            }
            for row in canonical_rows
        ),
    )
    for split in SPLITS:
        write_tsv_atomic(
            paths[split],
            fieldnames,
            (row.values for row in partitions[split]),
        )
    return paths


def read_tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Generated TSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def validate_written_outputs(
    paths: dict[str, Path],
    canonical_fieldnames: Sequence[str],
    canonical_rows: Sequence[CanonicalRow],
    partitions: dict[str, list[CanonicalRow]],
    assignment: AssignmentResult,
) -> None:
    manifest_fields, manifest = read_tsv_rows(paths["manifest"])
    if manifest_fields != ["alignment_id", "leakage_group_id", "split"]:
        raise ValueError(f"Unexpected manifest columns: {manifest_fields}")
    if len(manifest) != len(canonical_rows):
        raise ValueError(
            f"Manifest row count {len(manifest)} != {len(canonical_rows)}"
        )
    if len({row["alignment_id"] for row in manifest}) != len(manifest):
        raise ValueError("Manifest contains duplicate alignment IDs")
    if set(row["split"] for row in manifest) != set(SPLITS):
        raise ValueError("Manifest does not contain exactly train/dev/test")
    expected_manifest = [
        {
            "alignment_id": row.alignment_id,
            "leakage_group_id": row.leakage_group_id,
            "split": assignment.group_to_split[row.leakage_group_id],
        }
        for row in canonical_rows
    ]
    for output_line, (written_row, expected_row) in enumerate(
        zip(manifest, expected_manifest), start=2
    ):
        if written_row != expected_row:
            differing_columns = [
                column
                for column in ("alignment_id", "leakage_group_id", "split")
                if written_row.get(column) != expected_row[column]
            ]
            raise ValueError(
                f"Manifest round-trip mismatch at output line {output_line}; "
                f"columns={differing_columns}, "
                f"expected_alignment_id={expected_row['alignment_id']!r}"
            )

    for split in SPLITS:
        fields, written = read_tsv_rows(paths[split])
        if fields != list(canonical_fieldnames):
            raise ValueError(f"{split} columns differ from canonical columns")
        expected = partitions[split]
        if len(written) != len(expected):
            raise ValueError(
                f"{split} round-trip row count {len(written)} != {len(expected)}"
            )
        for output_line, (written_row, expected_row) in enumerate(
            zip(written, expected), start=2
        ):
            if written_row != expected_row.values:
                differing_columns = [
                    column
                    for column in canonical_fieldnames
                    if written_row.get(column) != expected_row.values.get(column)
                ]
                raise ValueError(
                    f"{split} full-row round-trip mismatch at output line "
                    f"{output_line}; columns={differing_columns}, "
                    f"expected_alignment_id={expected_row.alignment_id!r}"
                )


def split_statistics(
    rows: Sequence[CanonicalRow],
) -> tuple[int, int, Counter[int], Counter[int]]:
    single_label = sum(len(row.labels) == 1 for row in rows)
    multi_label = len(rows) - single_label
    label_counts: Counter[int] = Counter()
    for row in rows:
        label_counts.update(row.labels)
    groups = Counter(row.leakage_group_id for row in rows)
    return single_label, multi_label, label_counts, groups


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_report(
    canonical_path: Path,
    canonical_rows: Sequence[CanonicalRow],
    partitions: dict[str, list[CanonicalRow]],
    assignment: AssignmentResult,
    validation: ValidationResult,
    paths: dict[str, Path],
) -> None:
    overall_counts: Counter[int] = Counter()
    for row in canonical_rows:
        overall_counts.update(row.labels)
    overall_prevalence = {
        label: overall_counts[label] / len(canonical_rows)
        for label in PERMITTED_LABELS
    }

    per_split = {}
    for split in SPLITS:
        single, multi, counts, groups = split_statistics(partitions[split])
        per_split[split] = {
            "single": single,
            "multi": multi,
            "counts": counts,
            "groups": groups,
        }

    print("XED ENGLISH-TURKISH FIXED SPLIT")
    print(f"Python version: {platform.python_version()}")
    print(f"seed: {SEED}")
    print("algorithm: capacity-constrained group-aware multi-label greedy assignment")
    print(
        "           + deterministic equal-size cross-split swap hill climbing"
    )
    print(f"local improvement proposals: {LOCAL_IMPROVEMENT_PROPOSALS}")
    print(f"accepted improving swaps: {assignment.accepted_swaps}")
    print(
        "objective: sum over splits and labels of "
        "(actual positives - proportional target positives)^2 / target positives"
    )
    print(f"initial objective: {assignment.initial_objective:.12f}")
    print(f"final objective: {assignment.final_objective:.12f}")
    print("row-count constraint: hard exact capacities; leakage groups indivisible")
    print("movie grouping: not used")
    print()

    print("SPLIT SUMMARIES")
    for split in SPLITS:
        rows = partitions[split]
        stats = per_split[split]
        print(f"  {split}")
        print(
            f"    rows: {len(rows)} ({100 * len(rows) / len(canonical_rows):.4f}%)"
        )
        print(f"    leakage groups: {len(stats['groups'])}")
        print(f"    single-label rows: {stats['single']}")
        print(f"    multi-label rows: {stats['multi']}")
        print("    emotions:")
        for label in PERMITTED_LABELS:
            count = stats["counts"][label]
            prevalence = 100 * count / len(rows)
            print(
                f"      {label} {LABEL_NAMES[label]}: "
                f"{count} ({prevalence:.4f}%)"
            )
    print()

    print("STRATIFICATION QUALITY")
    print(
        "emotion\toverall %\ttrain %\ttrain abs dev pp\t"
        "dev %\tdev abs dev pp\ttest %\ttest abs dev pp"
    )
    maximum_deviation = 0.0
    for label in PERMITTED_LABELS:
        overall_percent = 100 * overall_prevalence[label]
        cells = [f"{label} {LABEL_NAMES[label]}", f"{overall_percent:.4f}"]
        for split in SPLITS:
            split_percent = (
                100
                * per_split[split]["counts"][label]
                / len(partitions[split])
            )
            deviation = abs(split_percent - overall_percent)
            maximum_deviation = max(maximum_deviation, deviation)
            cells.extend((f"{split_percent:.4f}", f"{deviation:.4f}"))
        print("\t".join(cells))
    print(
        "maximum absolute prevalence deviation across all labels/splits: "
        f"{maximum_deviation:.4f} percentage points"
    )
    print()

    print("LEAKAGE VALIDATION")
    print(f"  alignment overlap: {validation.alignment_overlap}")
    print(f"  leakage-group overlap: {validation.leakage_group_overlap}")
    print(
        "  exact-English-text cross-split overlap: "
        f"{validation.english_text_overlap}"
    )
    print(
        "  exact-Turkish-text cross-split overlap: "
        f"{validation.turkish_text_overlap}"
    )
    print(f"  source-key cross-split overlap: {validation.source_key_overlap}")
    print("  canonical alignments appearing exactly once: PASS")
    print("  all labels valid and present in every split: PASS")
    print("  missing texts or labels: 0")
    print()

    print("SHA-256")
    print(f"  {canonical_path.name}: {sha256(canonical_path)}")
    for name in ("manifest", "train", "dev", "test"):
        print(f"  {paths[name].name}: {sha256(paths[name])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_rows, fieldnames = read_canonical(args.input)
    validate_canonical(canonical_rows)
    groups = build_groups(canonical_rows)
    assignment = assign_groups(groups, len(canonical_rows))
    partitions = rows_by_split(canonical_rows, assignment)
    validation = validate_split(canonical_rows, partitions)
    paths = write_outputs(
        args.output_dir,
        fieldnames,
        canonical_rows,
        partitions,
        assignment,
    )
    validate_written_outputs(
        paths,
        fieldnames,
        canonical_rows,
        partitions,
        assignment,
    )
    print_report(
        args.input,
        canonical_rows,
        partitions,
        assignment,
        validation,
        paths,
    )


if __name__ == "__main__":
    main()
