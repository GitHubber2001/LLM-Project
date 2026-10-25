# Local model execution runbook

Run these commands from the repository root only after the required checkpoints and compatible model libraries have been supplied locally. The project commands do not download models or packages.

## 1. Register exact local checkpoints

```bash
cp configs/models.local.example.json configs/models.local.json
```

Edit `configs/models.local.json` so both entries contain exact checkpoint IDs, paths inside this repository, model sizes, multilingual-suitability notes, licences, accessibility notes, and selection justifications based on the actual local checkpoints. Do not use remote model IDs as paths.

## 2. Smoke-test both local tokenizer chat templates

```bash
python3 scripts/smoke_test_chat_templates.py
```

This loads tokenizers only, performs no model inference, and writes the Git-ignored `configs/chat_templates.local.json`. It tests both zero-shot and five-shot message structures for Qwen and Gemma. A system-less `user_preamble` format is used only if the exact tokenizer rejects the standard system role and the fallback passes locally.

## 3. Run the controlled prompt-sensitivity comparison on DEV

Create the four planned Qwen zero-shot DEV results:

```bash
python3 scripts/run_prompting.py --family qwen --language en --shots 0 --prompt-variant definitions_v1 --split dev --output results/predictions/prompt_sensitivity_definitions_v1_en_dev.jsonl
python3 scripts/run_prompting.py --family qwen --language tr --shots 0 --prompt-variant definitions_v1 --split dev --output results/predictions/prompt_sensitivity_definitions_v1_tr_dev.jsonl
python3 scripts/run_prompting.py --family qwen --language en --shots 0 --prompt-variant compact_v1 --split dev --output results/predictions/prompt_sensitivity_compact_v1_en_dev.jsonl
python3 scripts/run_prompting.py --family qwen --language tr --shots 0 --prompt-variant compact_v1 --split dev --output results/predictions/prompt_sensitivity_compact_v1_tr_dev.jsonl
```

Evaluate them:

```bash
python3 scripts/evaluate_predictions.py --predictions results/predictions/prompt_sensitivity_definitions_v1_en_dev.jsonl --experiment-id prompt_sensitivity_definitions_v1_en_dev
python3 scripts/evaluate_predictions.py --predictions results/predictions/prompt_sensitivity_definitions_v1_tr_dev.jsonl --experiment-id prompt_sensitivity_definitions_v1_tr_dev
python3 scripts/evaluate_predictions.py --predictions results/predictions/prompt_sensitivity_compact_v1_en_dev.jsonl --experiment-id prompt_sensitivity_compact_v1_en_dev
python3 scripts/evaluate_predictions.py --predictions results/predictions/prompt_sensitivity_compact_v1_tr_dev.jsonl --experiment-id prompt_sensitivity_compact_v1_tr_dev
```

Apply the planned DEV selection rule:

```bash
python3 scripts/select_prompt_variant.py \
  --metrics results/metrics/prompt_sensitivity_definitions_v1_en_dev_overall.json \
  --metrics results/metrics/prompt_sensitivity_definitions_v1_tr_dev_overall.json \
  --metrics results/metrics/prompt_sensitivity_compact_v1_en_dev_overall.json \
  --metrics results/metrics/prompt_sensitivity_compact_v1_tr_dev_overall.json
```

This writes the selection file and records the selected variant in `configs/prompting.json`. TEST data is neither accepted nor read.

## 4. Run all prompting conditions on DEV

```bash
python3 scripts/run_prompting.py --family qwen --language en --shots 0 --split dev --output results/predictions/qwen_zero_shot_en_dev.jsonl
python3 scripts/run_prompting.py --family qwen --language tr --shots 0 --split dev --output results/predictions/qwen_zero_shot_tr_dev.jsonl
python3 scripts/run_prompting.py --family qwen --language en --shots 5 --split dev --output results/predictions/qwen_five_shot_en_dev.jsonl
python3 scripts/run_prompting.py --family qwen --language tr --shots 5 --split dev --output results/predictions/qwen_five_shot_tr_dev.jsonl
python3 scripts/run_prompting.py --family gemma --language en --shots 0 --split dev --output results/predictions/gemma_zero_shot_en_dev.jsonl
python3 scripts/run_prompting.py --family gemma --language tr --shots 0 --split dev --output results/predictions/gemma_zero_shot_tr_dev.jsonl
python3 scripts/run_prompting.py --family gemma --language en --shots 5 --split dev --output results/predictions/gemma_five_shot_en_dev.jsonl
python3 scripts/run_prompting.py --family gemma --language tr --shots 5 --split dev --output results/predictions/gemma_five_shot_tr_dev.jsonl
```

Evaluate any DEV file with the evaluator, using a DEV-specific experiment ID. For example:

```bash
python3 scripts/evaluate_predictions.py --predictions results/predictions/qwen_zero_shot_en_dev.jsonl --experiment-id qwen_zero_shot_en_dev
```

Do not update `results/master_results.csv` with DEV metrics.

## 5. Train the three Qwen LoRA adapters

```bash
python3 scripts/prepare_peft_data.py
python3 scripts/train_peft.py --config configs/peft_english.json
python3 scripts/train_peft.py --config configs/peft_turkish.json
python3 scripts/train_peft.py --config configs/peft_bilingual.json
```

Training selects the lowest DEV-loss checkpoint and writes each final adapter to its condition's `best_adapter` directory.

## 6. Evaluate the adapters on DEV

```bash
python3 scripts/run_peft_inference.py --condition english_peft --language en --split dev --adapter results/peft/checkpoints/english_peft/best_adapter --output results/predictions/qwen_english_peft_en_dev.jsonl
python3 scripts/run_peft_inference.py --condition english_peft --language tr --split dev --adapter results/peft/checkpoints/english_peft/best_adapter --output results/predictions/qwen_english_peft_tr_dev.jsonl
python3 scripts/run_peft_inference.py --condition turkish_peft --language tr --split dev --adapter results/peft/checkpoints/turkish_peft/best_adapter --output results/predictions/qwen_turkish_peft_tr_dev.jsonl
python3 scripts/run_peft_inference.py --condition bilingual_peft --language en --split dev --adapter results/peft/checkpoints/bilingual_peft/best_adapter --output results/predictions/qwen_bilingual_peft_en_dev.jsonl
python3 scripts/run_peft_inference.py --condition bilingual_peft --language tr --split dev --adapter results/peft/checkpoints/bilingual_peft/best_adapter --output results/predictions/qwen_bilingual_peft_tr_dev.jsonl
```

Evaluate these files as needed with DEV-specific experiment IDs. Make every prompt, decoding, and checkpoint-selection decision before continuing.

## 7. Lock the final configuration once

```bash
python3 scripts/freeze_final_config.py --confirm-dev-decisions-complete
```

This records the model choices, verified chat-template modes and hashes, selected prompt and DEV-selection hash, TEST and demonstration hashes, and adapter tree hashes. Once the status is `FROZEN`, do not change these files or settings.

## 8. Run all thirteen TEST evaluations once

Prompting:

```bash
python3 scripts/run_prompting.py --family qwen --language en --shots 0 --split test --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_zero_shot_en_test.jsonl
python3 scripts/run_prompting.py --family qwen --language tr --shots 0 --split test --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_zero_shot_tr_test.jsonl
python3 scripts/run_prompting.py --family qwen --language en --shots 5 --split test --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_five_shot_en_test.jsonl
python3 scripts/run_prompting.py --family qwen --language tr --shots 5 --split test --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_five_shot_tr_test.jsonl
python3 scripts/run_prompting.py --family gemma --language en --shots 0 --split test --final-config configs/final_experiment_config.yaml --output results/predictions/gemma_zero_shot_en_test.jsonl
python3 scripts/run_prompting.py --family gemma --language tr --shots 0 --split test --final-config configs/final_experiment_config.yaml --output results/predictions/gemma_zero_shot_tr_test.jsonl
python3 scripts/run_prompting.py --family gemma --language en --shots 5 --split test --final-config configs/final_experiment_config.yaml --output results/predictions/gemma_five_shot_en_test.jsonl
python3 scripts/run_prompting.py --family gemma --language tr --shots 5 --split test --final-config configs/final_experiment_config.yaml --output results/predictions/gemma_five_shot_tr_test.jsonl
```

Qwen LoRA:

```bash
python3 scripts/run_peft_inference.py --condition english_peft --language en --split test --adapter results/peft/checkpoints/english_peft/best_adapter --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_english_peft_en_test.jsonl
python3 scripts/run_peft_inference.py --condition english_peft --language tr --split test --adapter results/peft/checkpoints/english_peft/best_adapter --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_english_peft_tr_test.jsonl
python3 scripts/run_peft_inference.py --condition turkish_peft --language tr --split test --adapter results/peft/checkpoints/turkish_peft/best_adapter --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_turkish_peft_tr_test.jsonl
python3 scripts/run_peft_inference.py --condition bilingual_peft --language en --split test --adapter results/peft/checkpoints/bilingual_peft/best_adapter --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_bilingual_peft_en_test.jsonl
python3 scripts/run_peft_inference.py --condition bilingual_peft --language tr --split test --adapter results/peft/checkpoints/bilingual_peft/best_adapter --final-config configs/final_experiment_config.yaml --output results/predictions/qwen_bilingual_peft_tr_test.jsonl
```

Partial TEST runs are rejected. Do not redesign the experiment after reading these outputs.

## 9. Score TEST and update the main results table

The prediction filenames use the master experiment IDs plus `_test.jsonl`, so this exact zsh loop scores and records all thirteen runs:

```bash
for experiment_id in qwen_zero_shot_en qwen_zero_shot_tr qwen_five_shot_en qwen_five_shot_tr gemma_zero_shot_en gemma_zero_shot_tr gemma_five_shot_en gemma_five_shot_tr qwen_english_peft_en qwen_english_peft_tr qwen_turkish_peft_tr qwen_bilingual_peft_en qwen_bilingual_peft_tr
do
  python3 scripts/evaluate_predictions.py --predictions "results/predictions/${experiment_id}_test.jsonl" --experiment-id "${experiment_id}"
  python3 scripts/update_master_results.py --metrics "results/metrics/${experiment_id}_overall.json"
done
```

## 10. Prepare candidates for human error review

For the required zero-shot versus English-adapted Turkish comparison:

```bash
python3 scripts/analyze_errors.py --baseline results/predictions/qwen_zero_shot_tr_test.jsonl --comparison results/predictions/qwen_english_peft_tr_test.jsonl --output results/analysis/qwen_zero_vs_english_peft_tr.csv
```

Complete the columns about possible causes manually. Update the report and presentation only from the saved predictions, metrics, main results table, and reviewed analysis.
