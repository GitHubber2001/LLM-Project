# Results directory

Real model experiments have been completed in the current environment.

- `master_results.csv` contains the recorded metrics for all 13 final TEST evaluations.
- `predictions/` contains raw and parsed JSONL prediction records for the main experiments.
- `metrics/` contains overall JSON and per-label CSV files generated from real predictions.
- `analysis/` contains human-review candidate tables after predictions exist.
- `peft/` receives Qwen adapter checkpoints and run metadata.
- `figures/` is reserved for figures derived from real metrics.
- `prompting/prompt_sensitivity/` contains the four controlled DEV-only prompt sensitivity outputs and the selection artifact.
- The eight main zero-shot and five-shot DEV conditions for Qwen and Gemma have completed, and their metrics are saved in `metrics/`.
- The selected prompt variant is `definitions_v1`, selected using DEV results only.

Never add hand-written or placeholder numeric results to these directories. Update the master table only with `scripts/update_master_results.py` using a completed metrics artifact. Do not use TEST results for prompt or model selection.
