#!/usr/bin/env python3
"""Run one local-only Qwen/Gemma prompting condition.

Development runs use ``data/splits/dev.tsv``. Test runs additionally require a
frozen final configuration whose model, prompt, demonstrations, parser, and
test-data hash match the requested run. Model loading always uses
``local_files_only=True`` and paths outside this repository are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.constants import (  # noqa: E402
    LANGUAGE_COLUMNS,
    MODEL_FAMILIES,
    PARSER_VERSION,
    PROMPT_VERSION,
    SEED,
)
from xed_project.chat_templates import (  # noqa: E402
    adapt_messages_for_chat_template,
    chat_template_sha256,
    verified_chat_template_mode,
)
from xed_project.io_utils import read_tsv  # noqa: E402
from xed_project.parsing import parse_prediction  # noqa: E402
from xed_project.prompting import (  # noqa: E402
    PROMPT_INSTRUCTIONS,
    build_messages,
    demonstrations_from_config,
)


DEFAULT_MODEL_CONFIG = ROOT / "configs" / "models.local.json"
DEFAULT_PROMPT_CONFIG = ROOT / "configs" / "prompting.json"
DEFAULT_DEMONSTRATIONS = ROOT / "configs" / "five_shot_examples.json"
DEFAULT_CHAT_TEMPLATE_CONFIG = ROOT / "configs" / "chat_templates.local.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def resolve_workspace_path(raw_path: object, *, description: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Missing {description}; configure a local workspace path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"{description} must remain inside the project workspace")
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def load_model_entry(path: Path, family: str) -> tuple[str, Path]:
    config = load_json(path)
    entry = config.get(family)
    if not isinstance(entry, dict):
        raise ValueError(f"Model config has no {family!r} entry")
    checkpoint_id = entry.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError(f"Exact {family} checkpoint_id is not configured")
    checkpoint_path = resolve_workspace_path(
        entry.get("checkpoint_path"), description=f"{family} checkpoint"
    )
    return checkpoint_id, checkpoint_path


def choose_prompt_variant(
    prompt_config: dict[str, object], requested: str | None, split: str
) -> str:
    candidates = prompt_config.get("candidate_variants")
    if not isinstance(candidates, list) or set(candidates) != set(PROMPT_INSTRUCTIONS):
        raise ValueError("Prompt config candidates do not match implemented variants")
    selected = prompt_config.get("selected_variant")
    if selected is not None and selected not in candidates:
        raise ValueError("Prompt config selected_variant is invalid")
    if requested is not None and requested not in candidates:
        raise ValueError(f"Unsupported prompt variant: {requested}")
    if split == "test":
        if prompt_config.get("status") != "DEV_SELECTED" or not isinstance(selected, str):
            raise ValueError("TEST is locked until a prompt variant is selected on DEV")
        if requested is not None and requested != selected:
            raise ValueError("Requested TEST prompt differs from the DEV-selected variant")
        return selected
    if requested is not None:
        return requested
    if isinstance(selected, str):
        return selected
    raise ValueError(
        "DEV prompt selection is pending; pass --prompt-variant explicitly"
    )


def load_chat_template_record(path: Path, family: str) -> dict[str, object]:
    config = load_json(path)
    if config.get("status") != "PASS":
        raise ValueError("Chat-template smoke-test artifact is not marked PASS")
    families = config.get("families")
    record = families.get(family) if isinstance(families, dict) else None
    if not isinstance(record, dict):
        raise ValueError(f"No chat-template smoke-test record for {family}")
    return record


def load_demonstrations(path: Path, language: str):
    config = load_json(path)
    examples = config.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Five-shot config is missing examples")
    return demonstrations_from_config(examples, language), config


def assert_frozen_test_config(
    path: Path,
    family: str,
    checkpoint_id: str,
    checkpoint_path: Path,
    demonstrations_config: dict[str, object],
    decoding: dict[str, object],
    prompt_variant: str,
    prompt_config: dict[str, object],
    chat_record: dict[str, object],
    test_path: Path,
) -> None:
    config = load_json(path)
    if config.get("status") != "FROZEN":
        raise ValueError("TEST is locked: final configuration status is not FROZEN")
    if config.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Frozen prompt version does not match code")
    if config.get("prompt_variant") != prompt_variant:
        raise ValueError("Frozen prompt variant differs from the DEV-selected variant")
    selection_value = prompt_config.get("selection_artifact")
    selection_path = resolve_workspace_path(
        selection_value, description="prompt-selection artifact"
    )
    if config.get("prompt_selection_sha256") != sha256(selection_path):
        raise ValueError("Frozen prompt-selection hash differs from local artifact")
    if config.get("decoding") != decoding:
        raise ValueError("Frozen decoding settings do not match prompt config")
    parser = config.get("parser")
    if not isinstance(parser, dict) or parser.get("version") != PARSER_VERSION:
        raise ValueError("Frozen parser version does not match code")
    models = config.get("models")
    entry = models.get(family) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"Frozen config lacks {family} model")
    if entry.get("checkpoint_id") != checkpoint_id:
        raise ValueError("Frozen and requested checkpoint identifiers differ")
    frozen_model_path = resolve_workspace_path(
        entry.get("checkpoint_path"), description=f"frozen {family} checkpoint"
    )
    if frozen_model_path != checkpoint_path:
        raise ValueError("Frozen and requested checkpoint paths differ")
    if entry.get("chat_template_mode") != chat_record.get("mode") or entry.get(
        "chat_template_sha256"
    ) != chat_record.get("chat_template_sha256"):
        raise ValueError("Frozen chat-template verification differs from local record")
    examples = demonstrations_config.get("examples")
    if not isinstance(examples, list):
        raise ValueError("Demonstration config has no examples")
    ids = [example.get("alignment_id") for example in examples]
    if config.get("demonstration_alignment_ids") != ids:
        raise ValueError("Frozen demonstration IDs do not match five-shot config")
    data = config.get("data")
    if not isinstance(data, dict) or data.get("test_sha256") != sha256(test_path):
        raise ValueError("Frozen test-data hash does not match local test data")
    if data.get("five_shot_sha256") != sha256(DEFAULT_DEMONSTRATIONS):
        raise ValueError("Frozen five-shot hash does not match local config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=MODEL_FAMILIES, required=True)
    parser.add_argument("--language", choices=("en", "tr"), required=True)
    parser.add_argument("--shots", choices=(0, 5), type=int, required=True)
    parser.add_argument("--prompt-variant", choices=tuple(PROMPT_INSTRUCTIONS))
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--prompt-config", type=Path, default=DEFAULT_PROMPT_CONFIG)
    parser.add_argument("--demonstrations", type=Path, default=DEFAULT_DEMONSTRATIONS)
    parser.add_argument(
        "--chat-template-config", type=Path, default=DEFAULT_CHAT_TEMPLATE_CONFIG
    )
    parser.add_argument("--final-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Prediction output must remain inside the workspace")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {output}")

    prompt_config = load_json(args.prompt_config)
    prompt_config_hash = sha256(args.prompt_config)
    if prompt_config.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Prompt config/code version mismatch")
    if prompt_config.get("parser_version") != PARSER_VERSION:
        raise ValueError("Parser config/code version mismatch")
    decoding = prompt_config.get("decoding")
    if not isinstance(decoding, dict):
        raise ValueError("Prompt config is missing decoding settings")
    prompt_variant = choose_prompt_variant(
        prompt_config, args.prompt_variant, args.split
    )

    checkpoint_id, checkpoint_path = load_model_entry(args.model_config, args.family)
    chat_record = load_chat_template_record(args.chat_template_config, args.family)
    demonstrations = []
    demo_config: dict[str, object] = load_json(args.demonstrations)
    if args.shots == 5:
        demonstrations, demo_config = load_demonstrations(
            args.demonstrations, args.language
        )

    split_path = ROOT / "data" / "splits" / f"{args.split}.tsv"
    if args.split == "test":
        if args.limit is not None:
            raise ValueError("Partial TEST runs are prohibited")
        if args.final_config is None:
            raise ValueError("TEST requires --final-config")
        assert_frozen_test_config(
            args.final_config,
            args.family,
            checkpoint_id,
            checkpoint_path,
            demo_config,
            decoding,
            prompt_variant,
            prompt_config,
            chat_record,
            split_path,
        )

    # Enforce offline behavior before importing model libraries.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Prompting requires locally installed torch and transformers; "
            "no package was downloaded"
        ) from error

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path), local_files_only=True, trust_remote_code=False
    )
    chat_template_mode = verified_chat_template_mode(
        chat_record,
        family=args.family,
        checkpoint_id=checkpoint_id,
        checkpoint_path=str(checkpoint_path.relative_to(ROOT)),
        template_sha256=chat_template_sha256(tokenizer.chat_template),
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="auto",
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    _, rows = read_tsv(split_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    text_column = LANGUAGE_COLUMNS[args.language]
    condition = "zero_shot" if args.shots == 0 else "five_shot"
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists() and not args.overwrite:
        raise FileExistsError(f"Partial output exists: {partial}")

    generation = {
        "do_sample": bool(decoding.get("do_sample", False)),
        "max_new_tokens": int(decoding.get("max_new_tokens", 64)),
        "num_beams": int(decoding.get("num_beams", 1)),
    }
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                messages = build_messages(
                    row[text_column],
                    args.language,
                    demonstrations,
                    prompt_variant=prompt_variant,
                )
                messages = adapt_messages_for_chat_template(
                    messages, chat_template_mode
                )
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(**inputs, **generation)
                prompt_length = inputs["input_ids"].shape[1]
                raw_output = tokenizer.decode(
                    output_ids[0][prompt_length:], skip_special_tokens=True
                ).strip()
                parsed = parse_prediction(raw_output)
                gold_labels = json.loads(row["labels"])
                record = {
                    "alignment_id": row["alignment_id"],
                    "language": args.language,
                    "model": args.family,
                    "checkpoint_id": checkpoint_id,
                    "condition": condition,
                    "split": args.split,
                    "input_text": row[text_column],
                    "gold_labels": gold_labels,
                    "raw_model_output": raw_output,
                    "parsed_predicted_labels": list(parsed.labels),
                    "malformed": parsed.malformed,
                    "parser_reason": parsed.reason,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_variant": prompt_variant,
                    "prompt_config_sha256": prompt_config_hash,
                    "chat_template_mode": chat_template_mode,
                    "chat_template_sha256": chat_record["chat_template_sha256"],
                    "parser_version": parsed.parser_version,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 50 == 0 or index == len(rows):
                    print(f"completed {index}/{len(rows)}")
        os.replace(partial, output)
    finally:
        if partial.exists() and output.exists():
            partial.unlink()
    print(f"predictions: {output}")


if __name__ == "__main__":
    main()
