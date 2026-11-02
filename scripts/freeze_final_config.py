#!/usr/bin/env python3
"""Freeze the final TEST configuration after all DEV decisions are complete.

The command requires explicit confirmation, two local model checkpoints, and
three completed local Qwen adapter directories. It records file hashes and
refuses paths outside the repository. Run it once; do not edit the frozen file
in response to TEST results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.chat_templates import CHAT_TEMPLATE_MODES  # noqa: E402
from xed_project.constants import PROMPT_VERSION  # noqa: E402
from xed_project.io_utils import write_json_atomic  # noqa: E402
from xed_project.prompting import PROMPT_INSTRUCTIONS  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def local_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {name}")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"{name} must remain inside the project workspace")
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Directory contains no files: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "final_experiment_config.yaml"
    )
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs" / "models.local.json"
    )
    parser.add_argument(
        "--prompt-config", type=Path, default=ROOT / "configs" / "prompting.json"
    )
    parser.add_argument(
        "--chat-template-config",
        type=Path,
        default=ROOT / "configs" / "chat_templates.local.json",
    )
    parser.add_argument(
        "--confirm-dev-decisions-complete", action="store_true", required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if config.get("status") == "FROZEN":
        raise ValueError("Configuration is already FROZEN")
    models_local = load_json(args.model_config)
    chat_templates = load_json(args.chat_template_config)
    if chat_templates.get("status") != "PASS":
        raise ValueError("Chat-template smoke tests are not complete")
    chat_families = chat_templates.get("families")
    if not isinstance(chat_families, dict):
        raise ValueError("Chat-template config has no families section")
    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("Final config has no models section")
    for family in ("qwen", "gemma"):
        local_entry = models_local.get(family)
        final_entry = models.get(family)
        if not isinstance(local_entry, dict) or not isinstance(final_entry, dict):
            raise ValueError(f"Missing {family} model configuration")
        checkpoint_id = local_entry.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError(f"Exact {family} checkpoint_id is missing")
        checkpoint_path = local_path(
            local_entry.get("checkpoint_path"), f"{family} checkpoint"
        )
        documentation_fields = (
            "model_size",
            "multilingual_suitability",
            "license",
            "accessibility",
            "selection_justification",
        )
        for field in documentation_fields:
            value = local_entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing {family} {field}")
            final_entry[field] = value
        chat_entry = chat_families.get(family)
        if not isinstance(chat_entry, dict) or chat_entry.get("status") != "PASS":
            raise ValueError(f"Missing passed {family} chat-template smoke test")
        if chat_entry.get("checkpoint_id") != checkpoint_id or chat_entry.get(
            "checkpoint_path"
        ) != str(checkpoint_path.relative_to(ROOT)):
            raise ValueError(f"{family} chat-template record does not match checkpoint")
        if chat_entry.get("mode") not in CHAT_TEMPLATE_MODES:
            raise ValueError(f"Invalid {family} verified chat-template mode")
        template_hash = chat_entry.get("chat_template_sha256")
        if not isinstance(template_hash, str) or len(template_hash) != 64:
            raise ValueError(f"Invalid {family} chat-template hash")
        final_entry["checkpoint_id"] = checkpoint_id
        final_entry["checkpoint_path"] = str(checkpoint_path.relative_to(ROOT))
        final_entry["chat_template_mode"] = chat_entry.get("mode")
        final_entry["chat_template_sha256"] = chat_entry.get("chat_template_sha256")

    prompt_config = load_json(args.prompt_config)
    if prompt_config.get("status") != "DEV_SELECTED":
        raise ValueError("Prompt sensitivity analysis is not completed on DEV")
    if prompt_config.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Prompt config/code version mismatch")
    decoding = prompt_config.get("decoding")
    if not isinstance(decoding, dict):
        raise ValueError("Prompt config has no decoding settings")
    selected_variant = prompt_config.get("selected_variant")
    if selected_variant not in PROMPT_INSTRUCTIONS:
        raise ValueError("Prompt config has no valid DEV-selected variant")
    selection_path = local_path(
        prompt_config.get("selection_artifact"), "prompt-selection artifact"
    )
    selection = load_json(selection_path)
    if (
        selection.get("status") != "COMPLETED"
        or selection.get("selected_variant") != selected_variant
        or selection.get("split") != "dev"
    ):
        raise ValueError("Prompt-selection artifact is incomplete or inconsistent")
    config["prompt_version"] = PROMPT_VERSION
    config["prompt_variant"] = selected_variant
    config["prompt_selection_sha256"] = sha256_file(selection_path)
    config["decoding"] = decoding

    adapters = config.get("adapters")
    if not isinstance(adapters, dict):
        raise ValueError("Final config has no adapters section")
    for condition in ("english_peft", "turkish_peft", "bilingual_peft"):
        run_dir = ROOT / "results" / "peft" / "checkpoints" / condition
        metadata_path = run_dir / "run_metadata.json"
        metadata = load_json(metadata_path)
        if metadata.get("status") != "COMPLETED":
            raise ValueError(f"Adapter run is not completed: {condition}")
        adapter_path = local_path(str(run_dir / "best_adapter"), f"{condition} adapter")
        adapters[condition] = {
            "path": str(adapter_path.relative_to(ROOT)),
            "sha256": sha256_tree(adapter_path),
        }

    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("Final config has no data section")
    test_path = local_path(data.get("test_path"), "test data")
    demos_path = local_path(data.get("five_shot_path"), "five-shot config")
    data["test_sha256"] = sha256_file(test_path)
    data["five_shot_sha256"] = sha256_file(demos_path)
    demo_config = load_json(demos_path)
    examples = demo_config.get("examples")
    if not isinstance(examples, list) or len(examples) != 5:
        raise ValueError("Five-shot config must contain exactly five examples")
    ids = [example.get("alignment_id") for example in examples]
    if config.get("demonstration_alignment_ids") != ids:
        raise ValueError("Draft demonstration IDs do not match five-shot artifact")

    config["status"] = "FROZEN"
    config["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["note"] = (
        "Frozen after DEV-only decisions. Do not modify in response to TEST results."
    )
    write_json_atomic(args.config, config)
    print(f"FROZEN: {args.config}")


if __name__ == "__main__":
    main()
