# XED English–Turkish Multi-Label Emotion Classification

University of Groningen — Introduction to Large Language Models - Group 7

## Project Overview

This project investigates cross-lingual multi-label emotion classification using the XED multilingual emotion dataset.

We compare zero-shot prompting, five-shot prompting, and parameter-efficient fine-tuning (LoRA) on aligned English and Turkish movie subtitles.

The primary model is Qwen2.5-3B-Instruct, while Gemma-2-2B-it is used as a secondary prompting model.

## Dataset

The dataset contains 6,931 aligned English–Turkish subtitle pairs.

| Partition   |  Examples |
|-------------|------:|
| Training    | 4,852 |
| Development | 1,386 |
| Test        | 693 |

The dataset includes 4,603 single-label and 2,328 multi-label examples.

The eight emotion labels are anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.

Turkish labels are reconstructed from English source annotations and projected onto aligned Turkish subtitles. They are not independent Turkish human annotations.

A fixed, group-aware multi-label stratification procedure was used with random seed 42. Exact-text and alignment-based leakage controls prevent overlapping examples across partitions.

See `docs/DATA_AUDIT.md` for class frequencies, split statistics, and validation results.

## Experiments

The project includes 13 final TEST evaluations:

| Model | Conditions |
| --- | --- |
| Qwen2.5-3B-Instruct | Zero-shot, five-shot, English-only LoRA, Turkish-only LoRA, bilingual LoRA |
| Gemma-2-2B-it | Zero-shot, five-shot |

Models are evaluated on the languages specified in the course experimental matrix.

Five-shot demonstrations are selected exclusively from the training set. The development set is used for model configuration and prompt selection. The test set is reserved for final evaluation.

## Repository Structure

- `configs/` — Experiment settings and fixed configurations
- `data/processed/` — Canonical aligned EN–TR dataset
- `data/splits/` — Fixed train/dev/test splits
- `data/peft/` — Generated PEFT training data
- `scripts/` — Data preparation, training, inference, and evaluation
- `src/xed_project/` — Shared Python modules
- `results/predictions/` — Saved model predictions
- `results/metrics/` — Overall and per-emotion evaluation results
- `results/peft/metadata/` — Recorded PEFT training metadata
- `results/analysis/` — Error analysis candidates
- `tests/` — Unit tests
- `docs/` — Data audit, workflow documentation, and reproduction instructions

The original EN–TR alignment and annotation sources are included under `subtitle-retrieval/`. The Turkish projection reference is located under `Projections/`.


## Environment and Dependencies

The prompting and PEFT experiments were run in separate environments.

### Prompting environment

- Python 3.11.3
- PyTorch 2.1.2
- Transformers 4.40.2

### PEFT training environment

- Python 3.9.25
- PyTorch 2.8.0+cu128
- Transformers 4.57.6
- PEFT 0.17.1
- Accelerate 1.10.1
- NVIDIA A100-PCIE-40GB

`requirements-model.txt` contains the pinned PEFT model dependencies. It does not reproduce the separate prompting environment.


### Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install model dependencies from a suitable package source:

```bash
python3 -m pip install -r requirements-model.txt
```

The model checkpoints must be obtained separately and placed in the repository at:

- `models/qwen/` — Qwen/Qwen2.5-3B-Instruct
- `models/gemma/` — google/gemma-2-2b-it

Gemma may require accepting its access terms.

Create the local model configuration:

```bash
cp configs/models.local.example.json configs/models.local.json
```

Verify that its checkpoint paths point to the correct locally available models.

The inference scripts load model checkpoints locally and do not automatically download them.

## Data Preparation

To regenerate the canonical dataset and fixed splits from the included source files:

```bash
python3 scripts/prepare_xed_tr.py
python3 scripts/split_xed_tr.py
```

To validate the dataset:

```bash
python3 scripts/validate_data.py
```

To prepare the six PEFT training and development files:

```bash
python3 scripts/prepare_peft_data.py
```

The generated files are written to `data/peft/`.

## Reproducing Experiments

Full execution instructions are available in `docs/RUNBOOK.md`.

The workflow covers:

1. Registering local model checkpoints.
2. Checking tokenizer and chat-template compatibility.
3. Selecting prompts using development data.
4. Running zero-shot and five-shot prompting.
5. Training English-only, Turkish-only, and bilingual LoRA adapters.
6. Evaluating the adapters on development data.
7. Freezing the configuration before final testing.
8. Running the final TEST experiments and computing metrics.

### Final Experiment Configurations

Two configuration files are provided:

- `configs/final_experiment_config.yaml` — Initial DRAFT configuration, intended to be frozen after completing a new reproduction run's development-stage decisions.
- `configs/final_experiment_config.frozen.yaml` — Archived FROZEN configuration recording the settings used for the reported experiments.

The archived configuration records model identifiers, prompt settings, adapter paths and hashes, selected demonstrations, and data hashes.

It does not contain model weights.

A fresh training run must use its own verified checkpoint and adapter information. Reproducing the archived inference conditions requires the corresponding original model and adapter artifacts.

## Evaluation and Results

The primary metrics are micro-averaged F1 and sample-averaged Jaccard index.

Additional metrics include macro F1, micro precision and recall, Hamming loss, exact-match ratio, per-emotion scores,