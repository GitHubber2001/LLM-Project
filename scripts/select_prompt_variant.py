#!/usr/bin/env python3
"""Select one prompt variant from controlled Qwen zero-shot DEV metrics."""

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

from xed_project.io_utils import write_json_atomic  # noqa: E402
from xed_project.prompting import PROMPT_INSTRUCTIONS  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the preregistered deterministic DEV ranking rule."""

    return sorted(
        summaries,
        key=lambda row: (
            -row["mean_micro_f1"],
            -row["mean_sample_jaccard"],
            row["mean_malformed_output_rate"],
            row["variant"],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", type=Path, action="append", required=True,
        help="Repeat exactly four times: two variants by English/Turkish DEV.",
    )
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "configs" / "prompt_sensitivity.json"
    )
    parser.add_argument(
        "--prompt-config", type=Path, default=ROOT / "configs" / "prompting.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "prompt_sensitivity" / "selection.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Selection output must remain inside the workspace")
    if output.exists():
        raise FileExistsError(f"Selection artifact already exists: {output}")

    plan = load_json(args.plan)
    if plan.get("split") != "dev":
        raise ValueError("Prompt sensitivity must use DEV only")
    variants = [entry.get("id") for entry in plan.get("variants", [])]
    if len(variants) != 2 or set(variants) - set(PROMPT_INSTRUCTIONS):
        raise ValueError("Plan must contain exactly the two implemented prompt variants")
    prompt_config = load_json(args.prompt_config)
    if prompt_config.get("status") != "DRAFT_DEV_SELECTION_PENDING":
        raise ValueError("Prompt config is not awaiting DEV selection")
    if set(prompt_config.get("candidate_variants", [])) != set(variants):
        raise ValueError("Prompt config candidates do not match sensitivity plan")
    expected_keys = {(variant, language) for variant in variants for language in ("en", "tr")}
    if len(args.metrics) != len(expected_keys):
        raise ValueError("Expected exactly four metrics files: two variants by two languages")

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    checkpoint_ids: set[str] = set()
    prompt_config_hashes: set[str] = set()
    for path in args.metrics:
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Metrics file must remain inside the workspace: {path}")
        metrics = load_json(path)
        if metrics.get("status") != "COMPLETED":
            raise ValueError(f"Metrics are not completed: {path}")
        if metrics.get("split") != "dev":
            raise ValueError(f"Non-DEV metrics are prohibited: {path}")
        if metrics.get("model") != "qwen" or metrics.get("condition") != "zero_shot":
            raise ValueError(f"Sensitivity metrics must be Qwen zero-shot DEV: {path}")
        variant = metrics.get("prompt_variant")
        language = metrics.get("language")
        key = (variant, language)
        if key not in expected_keys or key in rows:
            raise ValueError(f"Unexpected or duplicate variant/language metrics: {key}")
        for metric in ("micro_f1", "sample_jaccard", "malformed_output_rate"):
            if not isinstance(metrics.get(metric), (int, float)):
                raise ValueError(f"Missing numeric {metric}: {path}")
        checkpoint_id = metrics.get("checkpoint_id")
        prompt_config_hash = metrics.get("prompt_config_sha256")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError(f"Metrics lack checkpoint_id: {path}")
        if not isinstance(prompt_config_hash, str) or not prompt_config_hash:
            raise ValueError(f"Metrics lack prompt_config_sha256: {path}")
        checkpoint_ids.add(checkpoint_id)
        prompt_config_hashes.add(prompt_config_hash)
        rows[key] = metrics
    if set(rows) != expected_keys:
        raise ValueError(f"Missing sensitivity cells: {sorted(expected_keys - set(rows))}")
    if len(checkpoint_ids) != 1:
        raise ValueError("Prompt variants were not run with one exact Qwen checkpoint")
    if len(prompt_config_hashes) != 1:
        raise ValueError("Prompt variants did not share one fixed prompt/decoding config")

    summaries: list[dict[str, Any]] = []
    for variant in variants:
        language_rows = [rows[(variant, language)] for language in ("en", "tr")]
        summaries.append(
            {
                "variant": variant,
                "mean_micro_f1": sum(row["micro_f1"] for row in language_rows) / 2,
                "mean_sample_jaccard": sum(row["sample_jaccard"] for row in language_rows) / 2,
                "mean_malformed_output_rate": sum(
                    row["malformed_output_rate"] for row in language_rows
                ) / 2,
                "metric_files": [
                    str(path.resolve().relative_to(ROOT.resolve()))
                    for path in args.metrics
                    if load_json(path).get("prompt_variant") == variant
                ],
            }
        )
    summaries = rank_summaries(summaries)
    selected = summaries[0]["variant"]
    artifact = {
        "version": 1,
        "status": "COMPLETED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "model": "qwen",
        "condition": "zero_shot",
        "checkpoint_id": next(iter(checkpoint_ids)),
        "prompt_config_sha256": next(iter(prompt_config_hashes)),
        "languages": ["en", "tr"],
        "selection_rule": plan["selection"],
        "summaries": summaries,
        "selected_variant": selected,
        "plan_sha256": sha256(args.plan),
        "note": "Selected using DEV only. Freeze this choice before TEST.",
    }
    write_json_atomic(output, artifact)

    plan["status"] = "COMPLETED"
    plan["selected_variant"] = selected
    plan["selection_artifact"] = str(output.relative_to(ROOT))
    write_json_atomic(args.plan, plan)

    prompt_config["status"] = "DEV_SELECTED"
    prompt_config["selected_variant"] = selected
    prompt_config["selection_artifact"] = str(output.relative_to(ROOT))
    write_json_atomic(args.prompt_config, prompt_config)
    print(f"selected prompt variant: {selected}")
    print(f"selection artifact: {output}")


if __name__ == "__main__":
    main()
