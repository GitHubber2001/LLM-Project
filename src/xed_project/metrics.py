"""Dependency-free fixed-label multi-label evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .constants import LABELS, LABEL_SET, ZERO_DIVISION


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else ZERO_DIVISION


@dataclass(frozen=True)
class Evaluation:
    overall: dict[str, int | float | str]
    per_label: list[dict[str, int | float | str]]


def _as_label_set(values: Iterable[str], *, name: str) -> frozenset[str]:
    result = frozenset(values)
    invalid = result - LABEL_SET
    if invalid:
        raise ValueError(f"{name} contains invalid labels: {sorted(invalid)}")
    return result


def evaluate_label_sets(
    gold_sets: Sequence[Iterable[str]],
    predicted_sets: Sequence[Iterable[str]],
    malformed_flags: Sequence[bool] | None = None,
) -> Evaluation:
    if len(gold_sets) != len(predicted_sets):
        raise ValueError("Gold and prediction lengths differ")
    if not gold_sets:
        raise ValueError("Cannot evaluate an empty prediction file")
    if malformed_flags is not None and len(malformed_flags) != len(gold_sets):
        raise ValueError("Malformed-flag length differs from predictions")

    gold = [_as_label_set(values, name="gold") for values in gold_sets]
    predicted = [
        _as_label_set(values, name="prediction") for values in predicted_sets
    ]

    per_label: list[dict[str, int | float | str]] = []
    total_tp = total_fp = total_fn = 0
    for label in LABELS:
        tp = sum(label in g and label in p for g, p in zip(gold, predicted))
        fp = sum(label not in g and label in p for g, p in zip(gold, predicted))
        fn = sum(label in g and label not in p for g, p in zip(gold, predicted))
        support = sum(label in g for g in gold)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2 * precision * recall, precision + recall)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_label.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
            }
        )

    micro_precision = _ratio(total_tp, total_tp + total_fp)
    micro_recall = _ratio(total_tp, total_tp + total_fn)
    micro_f1 = _ratio(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )
    macro_f1 = sum(float(row["f1"]) for row in per_label) / len(LABELS)
    sample_jaccard = sum(
        1.0 if not (g | p) else len(g & p) / len(g | p)
        for g, p in zip(gold, predicted)
    ) / len(gold)
    hamming_loss = sum(
        len(g ^ p) for g, p in zip(gold, predicted)
    ) / (len(gold) * len(LABELS))
    exact_match = sum(g == p for g, p in zip(gold, predicted)) / len(gold)

    overall: dict[str, int | float | str] = {
        "num_examples": len(gold),
        "label_order": ",".join(LABELS),
        "zero_division": ZERO_DIVISION,
        "micro_f1": micro_f1,
        "sample_jaccard": sample_jaccard,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "hamming_loss": hamming_loss,
        "exact_match_ratio": exact_match,
    }
    if malformed_flags is not None:
        malformed_count = sum(bool(flag) for flag in malformed_flags)
        overall["malformed_count"] = malformed_count
        overall["malformed_output_rate"] = malformed_count / len(gold)
    return Evaluation(overall, per_label)
