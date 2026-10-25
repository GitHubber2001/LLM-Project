#!/usr/bin/env python3
"""Train a Qwen LoRA adapter from a local checkpoint and local JSONL data.

This script never downloads a model or package. It refuses non-Qwen configs,
non-LoRA methods, quantization, paths outside the project workspace, and any
training data marked as TEST.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xed_project.io_utils import read_jsonl, write_json_atomic  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def workspace_path(value: object, *, name: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {name}")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"{name} must remain inside the project workspace")
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def validate_config(config: dict[str, Any]) -> None:
    if config.get("model_family") != "qwen":
        raise ValueError("PEFT is permitted for Qwen only")
    if config.get("method") != "LoRA":
        raise ValueError("This local configuration implements LoRA only")
    if config.get("quantization") != "none":
        raise ValueError("Quantized training is not configured in this repository")
    lora = config.get("lora")
    if not isinstance(lora, dict) or lora.get("target_modules") != "all-linear":
        raise ValueError("LoRA target_modules must be the model-agnostic 'all-linear'")
    batch = config.get("per_device_batch_size")
    accumulation = config.get("gradient_accumulation_steps")
    if not isinstance(batch, int) or not isinstance(accumulation, int):
        raise ValueError("Invalid batch configuration")
    if config.get("effective_batch_size") != batch * accumulation:
        raise ValueError("Effective batch size does not match batch * accumulation")


def validate_records(records: list[dict[str, Any]], expected_split: str) -> None:
    if not records:
        raise ValueError(f"No {expected_split} records")
    for index, record in enumerate(records, start=1):
        if record.get("split") != expected_split:
            raise ValueError(f"Unexpected split in record {index}: {record.get('split')}")
        if str(record.get("split", "")).lower() == "test":
            raise ValueError("TEST data is prohibited in PEFT training/checkpoint selection")
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Invalid messages in record {index}")
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"Invalid message roles in record {index}: {roles}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs" / "models.local.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    validate_config(config)
    model_config = load_json(args.model_config)
    model_entry = model_config.get("qwen")
    if not isinstance(model_entry, dict):
        raise ValueError("models.local.json has no qwen entry")
    checkpoint_id = model_entry.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("Exact Qwen checkpoint_id is not configured")
    checkpoint_path = workspace_path(
        model_entry.get("checkpoint_path"), name="Qwen checkpoint"
    )
    train_path = workspace_path(config.get("train_data"), name="training data")
    dev_path = workspace_path(config.get("dev_data"), name="development data")
    output_dir = workspace_path(
        config.get("output_directory"), name="adapter output", must_exist=False
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Adapter output is not empty; pass --overwrite to continue: {output_dir}"
        )

    train_records = read_jsonl(train_path)
    dev_records = read_jsonl(dev_path)
    validate_records(train_records, "train")
    validate_records(dev_records, "dev")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
        from transformers import TrainingArguments
        import peft
    except ImportError as error:
        raise RuntimeError(
            "LoRA training requires locally installed torch, transformers, peft, "
            "and their local dependencies; nothing was downloaded"
        ) from error

    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype="auto",
    )
    model.config.use_cache = False
    lora_values = config["lora"]
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=lora_values["target_modules"],
        r=int(lora_values["rank"]),
        lora_alpha=int(lora_values["alpha"]),
        lora_dropout=float(lora_values["dropout"]),
        bias=str(lora_values["bias"]),
    )
    model = get_peft_model(model, lora_config)

    max_length = int(config["maximum_sequence_length"])

    class ConversationDataset(torch.utils.data.Dataset):
        def __init__(self, records: list[dict[str, Any]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            messages = self.records[index]["messages"]
            full_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=max_length,
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=max_length,
            )
            if not isinstance(full_ids, list) or not isinstance(prompt_ids, list):
                raise TypeError("Tokenizer chat template did not return token ID lists")
            labels = list(full_ids)
            for position in range(min(len(prompt_ids), len(labels))):
                labels[position] = -100
            if all(label == -100 for label in labels):
                raise ValueError("Assistant target was fully truncated; increase max length")
            return {
                "input_ids": list(full_ids),
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_batch_length = max(len(feature["input_ids"]) for feature in features)
        padded_ids: list[list[int]] = []
        padded_masks: list[list[int]] = []
        padded_labels: list[list[int]] = []
        for feature in features:
            padding = max_batch_length - len(feature["input_ids"])
            padded_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
            padded_masks.append(feature["attention_mask"] + [0] * padding)
            padded_labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(config["epochs"]),
        "per_device_train_batch_size": int(config["per_device_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "optim": str(config["optimizer"]),
        "lr_scheduler_type": str(config["scheduler"]),
        "logging_steps": 25,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "seed": seed,
        "data_seed": seed,
        "report_to": [],
        "remove_unused_columns": False,
    }
    argument_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in argument_names:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ConversationDataset(train_records),
        eval_dataset=ConversationDataset(dev_records),
        data_collator=collate,
    )
    train_result = trainer.train()
    best_dir = output_dir / "best_adapter"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple Metal Performance Shaders"
    else:
        device_name = "CPU"
    metadata = {
        "status": "COMPLETED",
        "base_model_checkpoint_id": checkpoint_id,
        "base_model_checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "condition": config["condition"],
        "configuration": config,
        "train_examples": len(train_records),
        "development_examples": len(dev_records),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_result.metrics,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "runtime": {
            "system": platform.system(),
            "machine": platform.machine(),
            "device": device_name,
        },
    }
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    print(f"best adapter: {best_dir}")
    print(f"metadata: {output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
