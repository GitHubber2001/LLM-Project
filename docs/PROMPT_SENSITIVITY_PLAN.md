# DEV-only prompt-sensitivity plan

Status: **BLOCKED** until an exact local Qwen checkpoint and a passed local tokenizer/chat-template smoke-test record are available.

## Controlled comparison

The experiment compares exactly two English instruction variants implemented in `src/xed_project/prompting.py`:

- `definitions_v1`: the current structured instruction with a full definition for each of the eight labels.
- `compact_v1`: shorter wording with concise definitions for the same eight labels.

Both variants describe the same multi-label task, use the same permitted English label names, require a JSON list, prohibit translated labels and explanatory text, use the same strict parser, and use deterministic decoding. Only the wording and formatting differ.

To keep the comparison small, both variants use the same Qwen checkpoint in the zero-shot condition on the complete fixed English and Turkish DEV partitions. No demonstrations are used, so the comparison focuses on prompt wording. TEST is never read.

## Metrics and selection rule

For each variant, calculate the mean across English and Turkish DEV for:

1. micro-F1;
2. sample-averaged Jaccard;
3. malformed-output rate.

Select the variant with the highest mean DEV micro-F1. Break a tie using higher mean DEV sample Jaccard, then lower mean malformed-output rate, and finally the alphabetically earlier variant ID. This rule is fixed in `configs/prompt_sensitivity.json` before any model run.

`scripts/select_prompt_variant.py` rejects non-DEV metrics, non-Qwen models, non-zero-shot conditions, missing language/variant cells, and non-numeric selection metrics. It writes `results/prompt_sensitivity/selection.json` and updates `configs/prompting.json` from `DRAFT_DEV_SELECTION_PENDING` to `DEV_SELECTED`.

## Required sequence

After local checkpoints are supplied:

1. Fill every field in `configs/models.local.json` from the actual checkpoints.
2. Run `scripts/smoke_test_chat_templates.py`; this loads tokenizers only and performs no model inference.
3. Run the four Qwen zero-shot DEV sensitivity cells listed in `docs/RUNBOOK.md`.
4. Evaluate those four DEV prediction files.
5. Run `scripts/select_prompt_variant.py` with the four metrics files.
6. Use the selected variant for the remaining DEV runs.
7. Lock the selected prompt, exact checkpoints, chat-template records, adapters, and other final settings before TEST.

No prompt may be selected or changed from TEST performance.
