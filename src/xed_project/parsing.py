"""Strict parsing for prompting outputs.

The parser accepts only a JSON list containing permitted English label names.
Case and surrounding whitespace are normalized, and duplicate valid labels are
removed. It never searches explanatory prose or maps synonyms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .constants import LABELS, LABEL_SET, PARSER_VERSION


@dataclass(frozen=True)
class ParseResult:
    labels: tuple[str, ...]
    malformed: bool
    reason: str | None
    parser_version: str = PARSER_VERSION


def parse_prediction(raw_output: str) -> ParseResult:
    """Parse one model response under the strict output contract.

    A well-formed empty JSON list is allowed and represents an empty predicted
    set. Any non-list JSON value, non-string member, or invalid label makes the
    whole response malformed and yields an empty set.
    """

    try:
        value = json.loads(raw_output.strip())
    except (json.JSONDecodeError, AttributeError) as error:
        reason = "invalid_json"
        if isinstance(error, AttributeError):
            reason = "non_string_output"
        return ParseResult((), True, reason)

    if not isinstance(value, list):
        return ParseResult((), True, "json_value_is_not_a_list")
    if not all(isinstance(item, str) for item in value):
        return ParseResult((), True, "list_contains_non_string_value")

    normalized = [item.strip().lower() for item in value]
    invalid = [item for item in normalized if item not in LABEL_SET]
    if invalid:
        return ParseResult((), True, "list_contains_invalid_label")

    present = set(normalized)
    return ParseResult(tuple(label for label in LABELS if label in present), False, None)

