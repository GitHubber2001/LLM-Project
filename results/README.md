# Results directory

No model experiment has been completed in the current local environment.

- `master_results.csv` is the authoritative experiment matrix. Its metric cells are empty while status is `BLOCKED`.
- `predictions/` receives raw and parsed JSONL prediction records.
- `metrics/` receives overall JSON and per-label CSV files from real predictions.
- `analysis/` receives human-review candidate tables after predictions exist.
- `peft/` receives Qwen adapter checkpoints and run metadata.
- `figures/` is reserved for figures derived from real metrics.
- `prompt_sensitivity/` receives the real DEV-only selection artifact after the four controlled sensitivity cells complete.

Never add hand-written or placeholder numeric results to these directories. Update the master table only with `scripts/update_master_results.py` using a completed metrics artifact.
