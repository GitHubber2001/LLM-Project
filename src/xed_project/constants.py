"""Project-wide constants that define the fixed experimental task."""

from __future__ import annotations

LABELS: tuple[str, ...] = (
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
)
LABEL_SET = frozenset(LABELS)
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
NUMERIC_TO_LABEL = {index + 1: label for index, label in enumerate(LABELS)}

LANGUAGE_COLUMNS = {"en": "english_text", "tr": "turkish_text"}
LANGUAGE_NAMES = {"en": "English", "tr": "Turkish"}
MODEL_FAMILIES = ("qwen", "gemma")
SEED = 42
PROMPT_VERSION = "xed-json-v1"
PARSER_VERSION = "strict-json-list-v1"
ZERO_DIVISION = 0.0

