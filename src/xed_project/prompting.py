"""Prompt construction shared by Qwen and Gemma experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .constants import LABELS, LANGUAGE_NAMES


DEFINITIONS_INSTRUCTION = """You are performing multi-label emotion classification for movie subtitle text.

Use only these eight English emotion labels:
- anger: strong displeasure, hostility, or rage
- anticipation: expectation or looking forward to a future event
- disgust: revulsion, strong dislike, or rejection
- fear: alarm, anxiety, or response to perceived danger
- joy: happiness, pleasure, or delight
- sadness: sorrow, grief, or unhappiness
- surprise: a reaction to something unexpected
- trust: confidence, acceptance, or reliance on someone or something

The text may express one or more emotions. Return only a valid JSON list containing one or more of the permitted English label names. Do not translate label names. Do not add explanations, Markdown, or any other text. Example: [\"joy\", \"trust\"]"""

COMPACT_INSTRUCTION = """Classify the emotions expressed by one movie subtitle. This is a multi-label task, so return one or more labels when appropriate.

The only permitted English labels are:
- anger: hostility, rage, or strong displeasure
- anticipation: expectation of a future event
- disgust: revulsion, rejection, or strong dislike
- fear: alarm, anxiety, or perceived danger
- joy: happiness, pleasure, or delight
- sadness: sorrow, grief, or unhappiness
- surprise: reaction to something unexpected
- trust: confidence, acceptance, or reliance

Return only a valid JSON list of permitted English label names, such as [\"joy\", \"trust\"]. Do not translate the labels or add prose, Markdown, or explanations."""

PROMPT_INSTRUCTIONS = {
    "definitions_v1": DEFINITIONS_INSTRUCTION,
    "compact_v1": COMPACT_INSTRUCTION,
}
DEFAULT_PROMPT_VARIANT = "definitions_v1"

# Backwards-compatible name used by the fixed PEFT-data preparation path. Prompt
# sensitivity applies to prompting experiments; PEFT instruction data retains this
# preregistered instruction unless its own DEV-controlled change is documented.
INSTRUCTION = PROMPT_INSTRUCTIONS[DEFAULT_PROMPT_VARIANT]


@dataclass(frozen=True)
class Demonstration:
    alignment_id: str
    text: str
    labels: tuple[str, ...]


def _validate_labels(labels: Iterable[str]) -> tuple[str, ...]:
    values = tuple(labels)
    if not values or any(label not in LABELS for label in values):
        raise ValueError(f"Invalid demonstration labels: {values!r}")
    if tuple(label for label in LABELS if label in set(values)) != values:
        raise ValueError(f"Labels must use fixed canonical order: {values!r}")
    return values


def build_messages(
    text: str,
    language: str,
    demonstrations: Sequence[Demonstration] = (),
    *,
    prompt_variant: str = DEFAULT_PROMPT_VARIANT,
) -> list[dict[str, str]]:
    """Build chat messages using the same English instruction for both inputs."""

    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language: {language!r}")
    if not text.strip():
        raise ValueError("Input text is empty")
    if len(demonstrations) not in (0, 5):
        raise ValueError("Prompting condition must use zero or exactly five examples")
    instruction = PROMPT_INSTRUCTIONS.get(prompt_variant)
    if instruction is None:
        raise ValueError(f"Unsupported prompt variant: {prompt_variant!r}")

    messages: list[dict[str, str]] = [{"role": "system", "content": instruction}]
    for example in demonstrations:
        labels = _validate_labels(example.labels)
        messages.append(
            {
                "role": "user",
                "content": f"{LANGUAGE_NAMES[language]} subtitle: {example.text}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(labels, ensure_ascii=False),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": f"{LANGUAGE_NAMES[language]} subtitle: {text}",
        }
    )
    return messages


def demonstrations_from_config(
    records: Sequence[Mapping[str, object]], language: str
) -> list[Demonstration]:
    if language not in ("en", "tr"):
        raise ValueError(f"Unsupported language: {language!r}")
    text_key = "english_text" if language == "en" else "turkish_text"
    result: list[Demonstration] = []
    for record in records:
        raw_labels = record.get("gold_labels")
        if not isinstance(raw_labels, list) or not all(
            isinstance(label, str) for label in raw_labels
        ):
            raise ValueError("Five-shot config contains invalid gold_labels")
        alignment_id = record.get("alignment_id")
        text = record.get(text_key)
        if not isinstance(alignment_id, str) or not isinstance(text, str):
            raise ValueError("Five-shot config is missing required text or alignment ID")
        result.append(Demonstration(alignment_id, text, tuple(raw_labels)))
    if len(result) != 5:
        raise ValueError(f"Expected exactly five demonstrations, found {len(result)}")
    return result

