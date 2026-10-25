"""Checkpoint-verified chat-template compatibility helpers."""

from __future__ import annotations

import hashlib
from typing import Mapping, Sequence


CHAT_TEMPLATE_MODES = ("system", "user_preamble")


def chat_template_sha256(template: object) -> str:
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Tokenizer has no non-empty chat_template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def adapt_messages_for_chat_template(
    messages: Sequence[Mapping[str, str]], mode: str
) -> list[dict[str, str]]:
    """Apply only a mode proven by the local tokenizer smoke test.

    ``user_preamble`` preserves the exact English instruction content but moves it
    into the first user message for templates that reject a system role.
    """

    copied = [dict(message) for message in messages]
    if not copied or copied[0].get("role") != "system":
        raise ValueError("Canonical messages must begin with one system instruction")
    if mode == "system":
        return copied
    if mode != "user_preamble":
        raise ValueError(f"Unsupported chat-template mode: {mode!r}")

    instruction = copied[0].get("content")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("System instruction is empty")
    remaining = copied[1:]
    first_user = next(
        (index for index, message in enumerate(remaining) if message.get("role") == "user"),
        None,
    )
    if first_user is None:
        raise ValueError("No user message is available for the instruction preamble")
    user_content = remaining[first_user].get("content")
    if not isinstance(user_content, str) or not user_content.strip():
        raise ValueError("First user message is empty")
    remaining[first_user] = {
        **remaining[first_user],
        "content": f"{instruction}\n\n{user_content}",
    }
    return remaining


def verified_chat_template_mode(
    record: object,
    *,
    family: str,
    checkpoint_id: str,
    checkpoint_path: str,
    template_sha256: str,
) -> str:
    if not isinstance(record, dict):
        raise ValueError(f"No chat-template smoke-test record for {family}")
    expected = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_path,
        "chat_template_sha256": template_sha256,
        "status": "PASS",
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"Chat-template smoke-test {field} mismatch for {family}: "
                f"{record.get(field)!r} != {value!r}"
            )
    mode = record.get("mode")
    if mode not in CHAT_TEMPLATE_MODES:
        raise ValueError(f"Invalid verified chat-template mode for {family}: {mode!r}")
    checks = record.get("checks")
    if not isinstance(checks, dict) or checks.get("zero_shot") != "PASS" or checks.get("five_shot") != "PASS":
        raise ValueError(f"Incomplete chat-template smoke checks for {family}")
    return str(mode)
