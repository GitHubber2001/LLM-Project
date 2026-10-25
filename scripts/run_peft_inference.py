#!/usr/bin/env python3
"""Generate local Qwen LoRA predictions for one language and split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.constants import (  # noqa: E402
    LANGUAGE_COLUMNS,
    PARSER_VERSION,
    PROMPT_VERSION,
    SEED,
)
from xed_project.io_utils import read_tsv  # noqa: E402
from xed_project.parsing import parse_prediction  # noqa: E402
from xed_project.prompting import build_messages  # noqa: E402


CONDITION_LANGUAGES = {
    "english_peft": ("en", "tr"),
    "turkish_peft": ("tr",),
    "bilingual_peft": ("en", "tr"),
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def workspace_path(value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
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


def sha256(path: Path) -> str:
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


def validate_test_lock(
    path: Path,
    condition: str,
    checkpoint_id: str,
    checkpoint_path: Path,
    adapter_path: Path,
    decoding: dict[str, Any],
    test_path: Path,
) -> None:
    config = load_json(path)
    if config.get("status") != "FROZEN":
        raise ValueError("TEST is locked: final configuration status is not FROZEN")
    models = config.get("models")
    qwen = models.get("qwen") if isinstance(models, dict) else None
    if not isinstance(qwen, dict):
        raise ValueError("Frozen config has no Qwen model")
    if qwen.get("checkpoint_id") != checkpoint_id:
        raise ValueError("Frozen Qwen identifier differs from requested model")
    if workspace_path(qwen.get("checkpoint_path"), "frozen Qwen checkpoint") != checkpoint_path:
        raise ValueError("Frozen Qwen path differs from requested model")
    adapters = config.get("adapters")
    entry = adapters.get(condition) if isinstance(adapters, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"Frozen config has no {condition} adapter")
    if workspace_path(entry.get("path"), "frozen adapter") != adapter_path:
        raise ValueError("Frozen adapter path differs from requested adapter")
    if entry.get("sha256") != sha256_tree(adapter_path):
        raise ValueError("Frozen adapter hash differs from requested adapter")
    if config.get("decoding") != decoding:
        raise ValueError("Frozen decoding settings do not match prompt config")
    data = config.get("data")
    if not isinstance(data, dict) or data.get("test_sha256") != sha256(test_path):
        raise ValueError("Frozen test-data hash differs from local data")
    parser = config.get("parser")
    if config.get("prompt_version") != PROMPT_VERSION or not isinstance(parser, dict) or parser.get("version") != PARSER_VERSION:
        raise ValueError("Frozen prompt/parser versions differ from code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITION_LANGUAGES), required=True)
    parser.add_argument("--language", choices=("en", "tr"), required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs" / "models.local.json"
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--prompt-config", type=Path, default=ROOT / "configs" / "prompting.json"
    )
    parser.add_argument("--final-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.language not in CONDITION_LANGUAGES[args.condition]:
        raise ValueError(
            f"{args.condition} is not a required evaluation for {args.language}"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Prediction output must remain inside the workspace")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output}")

    model_config = load_json(args.model_config)
    qwen = model_config.get("qwen")
    if not isinstance(qwen, dict):
        raise ValueError("Model config has no Qwen entry")
    checkpoint_id = qwen.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("Exact Qwen checkpoint_id is not configured")
    checkpoint_path = workspace_path(qwen.get("checkpoint_path"), "Qwen checkpoint")
    adapter_path = workspace_path(args.adapter, "LoRA adapter")
    run_metadata_path = adapter_path.parent / "run_metadata.json"
    if not run_metadata_path.exists():
        run_metadata_path = adapter_path.parent.parent / "run_metadata.json"
    run_metadata = load_json(run_metadata_path)
    if run_metadata.get("status") != "COMPLETED" or run_metadata.get("condition") != args.condition:
        raise ValueError("Adapter metadata does not match a completed requested condition")
    if run_metadata.get("base_model_checkpoint_id") != checkpoint_id:
        raise ValueError("Adapter and requested base checkpoint identifiers differ")

    prompt_config = load_json(args.prompt_config)
    decoding = prompt_config.get("decoding")
    if not isinstance(decoding, dict):
        raise ValueError("Prompt config has no decoding section")
    split_path = ROOT / "data" / "splits" / f"{args.split}.tsv"
    if args.split == "test":
        if args.limit is not None:
            raise ValueError("Partial TEST runs are prohibited")
        if args.final_config is None:
            raise ValueError("TEST requires --final-config")
        validate_test_lock(
            args.final_config,
            args.condition,
            checkpoint_id,
            checkpoint_path,
            adapter_path,
            decoding,
            split_path,
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "PEFT inference requires locally installed torch, transformers, and peft"
        ) from error

    random.seed(SEED)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path), local_files_only=True, trust_remote_code=False
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    generation = {
        "do_sample": bool(decoding.get("do_sample", False)),
        "max_new_tokens": int(decoding.get("max_new_tokens", 64)),
        "num_beams": int(decoding.get("num_beams", 1)),
    }
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                messages = build_messages(row[text_column], args.language)
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    generated = model.generate(**inputs, **generation)
                prompt_length = inputs["input_ids"].shape[1]
                raw = tokenizer.decode(
                    generated[0][prompt_length:], skip_special_tokens=True
                ).strip()
                parsed = parse_prediction(raw)
                record = {
                    "alignment_id": row["alignment_id"],
                    "language": args.language,
                    "model": "qwen",
                    "checkpoint_id": checkpoint_id,
                    "condition": args.condition,
                    "split": args.split,
                    "input_text": row[text_column],
                    "gold_labels": json.loads(row["labels"]),
                    "raw_model_output": raw,
                    "parsed_predicted_labels": list(parsed.labels),
                    "malformed": parsed.malformed,
                    "parser_reason": parsed.reason,
                    "prompt_version": PROMPT_VERSION,
                    "parser_version": PARSER_VERSION,
                    "adapter_path": str(adapter_path.relative_to(ROOT)),
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
