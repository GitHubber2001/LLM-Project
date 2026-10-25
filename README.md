# XED English-Turkish LLM Project

This folder contains the working code for the course project on English/Turkish multi-label emotion classification with XED.

## Current status

Completed:
- English-Turkish alignment: 6,931 examples
- fixed 70/20/10 TRAIN/DEV/TEST split with seed 42
- leakage validation
- five fixed TRAIN examples for five-shot prompting
- prompting, parsing, evaluation, and LoRA training code
- Qwen/Gemma experiment configs

Not completed yet:
- exact local Qwen and Gemma checkpoint selection
- DEV prompt comparison
- Qwen LoRA training
- final TEST model evaluation and metrics

## Folder structure

```text
configs/                 experiment settings
scripts/                 commands for data, prompting, training, and evaluation
src/xed_project/         shared Python code
tests/                   unit tests
data/processed/          aligned EN-TR dataset
data/splits/             fixed TRAIN/DEV/TEST files and manifest
data/peft/               generated LoRA training JSONL (create locally)
subtitle-retrieval/      minimum XED EN-TR source files needed by our preparation code
Projections/             Turkish XED projection file kept as source/reference
docs/                    walkthrough, data audit, runbook, prompt plan, course brief
results/                 experiment outputs and master result table
```

The full upstream XED repository is intentionally not included. Files for unrelated languages, the legacy BERT notebook, generated PEFT JSONL files, model weights, adapters, report drafts, and presentation drafts were left out.

## First checks

From the project root:

```bash
python3 scripts/validate_data.py --report docs/DATA_AUDIT.md
python3 -m unittest discover -s tests -v
```

To regenerate the six LoRA/PEFT JSONL files:

```bash
python3 scripts/prepare_peft_data.py
```

## Model experiments

Qwen is the primary model:
- zero-shot prompting
- five-shot prompting
- English-only LoRA
- Turkish-only LoRA
- bilingual LoRA

Gemma is the secondary model:
- zero-shot prompting
- five-shot prompting

The exact commands and DEV-before-TEST workflow are in `docs/RUNBOOK.md`.

Do not use TEST results to choose prompts or training settings. TRAIN is used for training/demonstrations, DEV for decisions, and TEST only for the final evaluation.

## Dataset note

Turkish XED labels are projected reference labels. The upstream XED attribution and licence information are preserved in `docs/UPSTREAM_XED_README.md`.
