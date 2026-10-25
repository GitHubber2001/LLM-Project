#!/usr/bin/env python3
"""Verify local Qwen/Gemma tokenizer chat templates without model inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.chat_templates import (  # noqa: E402
    CHAT_TEMPLATE_MODES,
    adapt_messages_for_chat_template,
    chat_template_sha256,
)
from xed_project.io_utils import write_json_atomic  # noqa: E402
from xed_project.prompting import (  # noqa: E402
    DEFAULT_PROMPT_VARIANT,
    Demonstration,
    PROMPT_INSTRUCTIONS,
    build_messages,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def workspace_checkpoint(value: object, family: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing local {family} checkpoint_path")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"{family} checkpoint must remain inside the workspace")
    if not path.is_dir():
        raise FileNotFoundError(f"{family} checkpoint directory does not exist: {path}")
    return path


def smoke_mode(tokenizer: Any, mode: str, prompt_variant: str) -> dict[str, str]:
    instruction = PROMPT_INSTRUCTIONS[prompt_variant]
    demonstration = Demonstration("smoke-id", "A short example.", ("joy",))
    cases = {
        "zero_shot": build_messages(
            "A tokenizer-only smoke test.", "en", prompt_variant=prompt_variant
        ),
        "five_shot": build_messages(
            "A tokenizer-only smoke test.",
            "en",
            [demonstration] * 5,
            prompt_variant=prompt_variant,
        ),
    }
    checks: dict[str, str] = {}
    for name, canonical_messages in cases.items():
        messages = adapt_messages_for_chat_template(canonical_messages, mode)
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if not isinstance(rendered, str) or instruction not in rendered:
            raise ValueError(f"{name}: rendered prompt lost the English instruction")
        if "A tokenizer-only smoke test." not in rendered:
            raise ValueError(f"{name}: rendered prompt lost the target subtitle")
        if not isinstance(token_ids, list) or not token_ids or not all(
            isinstance(value, int) for value in token_ids
        ):
            raise ValueError(f"{name}: tokenizer did not return a non-empty ID list")
        checks[name] = "PASS"
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs" / "models.local.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs" / "chat_templates.local.json",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=tuple(PROMPT_INSTRUCTIONS),
        default=DEFAULT_PROMPT_VARIANT,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Smoke-test output must remain inside the workspace")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")

    model_config = load_json(args.model_config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Tokenizer smoke tests require a locally installed Transformers package"
        ) from error

    result: dict[str, Any] = {
        "version": 1,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Tokenizer/chat-template verification only; no model inference.",
        "prompt_variant_used_for_smoke_test": args.prompt_variant,
        "transformers_version": transformers.__version__,
        "families": {},
    }
    failures: list[str] = []
    for family in ("qwen", "gemma"):
        entry = model_config.get(family)
        if not isinstance(entry, dict):
            raise ValueError(f"Model config has no {family} entry")
        checkpoint_id = entry.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError(f"Exact {family} checkpoint_id is not configured")
        checkpoint_path = workspace_checkpoint(entry.get("checkpoint_path"), family)
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_path), local_files_only=True, trust_remote_code=False
        )
        template_hash = chat_template_sha256(tokenizer.chat_template)
        attempts: dict[str, dict[str, Any]] = {}
        selected_mode: str | None = None
        selected_checks: dict[str, str] | None = None
        for mode in CHAT_TEMPLATE_MODES:
            try:
                checks = smoke_mode(tokenizer, mode, args.prompt_variant)
                attempts[mode] = {"status": "PASS"}
                if selected_mode is None:
                    selected_mode = mode
                    selected_checks = checks
            except Exception as error:  # record the local tokenizer rejection
                attempts[mode] = {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        if selected_mode is None or selected_checks is None:
            failures.append(family)
            family_status = "FAIL"
        else:
            family_status = "PASS"
        result["families"][family] = {
            "status": family_status,
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "tokenizer_class": type(tokenizer).__name__,
            "chat_template_sha256": template_hash,
            "mode": selected_mode,
            "checks": selected_checks,
            "attempts": attempts,
        }

    if failures:
        result["status"] = "FAIL"
        write_json_atomic(output, result)
        raise RuntimeError(
            "No verified chat-template mode for: "
            + ", ".join(failures)
            + f"; diagnostic artifact: {output}"
        )
    write_json_atomic(output, result)
    print(f"chat-template smoke test: PASS -> {output}")
    for family, record in result["families"].items():
        print(f"{family}: mode={record['mode']} tokenizer={record['tokenizer_class']}")


if __name__ == "__main__":
    main()
