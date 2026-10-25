# Project walkthrough

This document explains the project in plain language so that a student can understand the design, run the code, and defend the choices orally. For exact data counts and hashes, see `DATA_AUDIT.md`. For the current completion state, see `COURSE_REQUIREMENTS_CHECKLIST.md`.

## What problem does the project solve?

The project asks a model to read a short movie-subtitle line and identify every relevant emotion. The input can be English or Turkish, but the output vocabulary is always the same eight English names: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.

The central research question is not only “Can a model classify emotions?” It is also “How does performance change across languages and adaptation methods?” The experiment therefore compares:

- English input with Turkish input;
- zero-shot prompting with five-shot prompting;
- Qwen with Gemma for prompting;
- Qwen prompting with Qwen LoRA adaptation;
- English-only, Turkish-only, and bilingual Qwen adaptation.

No model result is available because the required local checkpoints are absent. The repository contains the data, prompt, parser, training, evaluation, and TEST-protection code needed to answer these questions once the checkpoints are supplied.

## What does multi-label mean?

In single-label classification, exactly one class is chosen. That would be wrong for these subtitles. A sentence such as a fearful warning may also express anger or anticipation. Multi-label classification represents the target as a set, for example:

```json
["fear", "anticipation"]
```

Every one of the eight labels is treated as its own yes/no decision. The number of predicted labels can vary between examples. The code never converts the target to one class and never uses a softmax-style “pick exactly one” evaluation.

## Why are there eight emotions?

XED uses eight core emotion categories: anger, anticipation, disgust, fear, joy, sadness, surprise, and trust. This inventory is fixed by the dataset and project brief. Its order is centralized in `src/xed_project/constants.py` and reused by data validation, parsing, training targets, and metrics. Fixing the order matters because every metric must compare the same eight binary decisions.

## What does XED contain?

XED is a multilingual dataset built from movie subtitles. The repository includes original annotation exports, projected-language TSV files, subtitle alignment pairs, and alignment metadata. The English annotations are linked to source subtitle document and segment identifiers. Other language versions were created by projecting information through aligned subtitle lines.

For this project, the important raw files are:

- `subtitle-retrieval/students/pairs-tr.txt`, containing aligned source and Turkish subtitle rows;
- `subtitle-retrieval/students/student_annotations.csv`, containing the source metadata used to reconstruct English emotion labels;
- `Projections/tr-projections2.tsv`, which is useful as a diagnostic but is not used as the primary label join.

The original Turkish pair file includes English-to-Turkish and Danish-to-Turkish records. The project keeps only the 6,931 English-to-Turkish rows.

## Why are English and Turkish paired?

Each retained row contains an English source subtitle and its aligned Turkish subtitle counterpart. Keeping the pair together makes two fair comparisons possible:

1. the model can classify the same underlying aligned example in either language;
2. the same reference label set can be used when comparing English and Turkish predictions.

The pair must remain together through splitting. If the English version appeared in TRAIN while the Turkish version appeared in TEST, a model could see almost the same example during training. That would overstate generalization.

## Where do the Turkish labels come from?

The Turkish texts were not independently labelled by Turkish annotators for this project. Labels are reconstructed from the English source annotation metadata and then associated with the aligned Turkish counterpart. They are therefore projected reference labels.

This wording is important. It is justified to say “the Turkish aligned row uses the projected reference labels.” It is not justified to say “Turkish annotators independently judged the sentence to have these emotions.” Projection can introduce error because subtitle alignment is not always a literal one-to-one translation and emotional nuance may change.

The audit also shows why the code does not look up labels by exact Turkish text alone. Of 6,536 rows that can be labelled by both exact Turkish lookup and source metadata, only 4,055 agree exactly (62.04%). Exact Turkish lookup also misses 395 retained alignments. Source document and segment IDs provide more reliable source information.

## What is an alignment ID?

An alignment ID is a stable identifier built from the English document and segment plus the Turkish document and segment. A simplified form is:

```text
english-document#english-segment=>turkish-document#turkish-segment
```

It identifies the pair, not just one sentence. Text alone is unsafe as an identifier because short subtitle phrases such as “Hey!” can repeat in many films and may carry different emotions in different contexts.

## What does the preprocessing pipeline do?

`scripts/prepare_xed_tr.py` builds the prepared aligned dataset:

1. Read the untouched Turkish subtitle-pair file.
2. Keep only rows whose source language is English and target language is Turkish.
3. Construct the alignment ID from document and segment fields.
4. Find every source segment in the annotation metadata.
5. Union all emotion annotations for that source segment into a sorted multi-label set.
6. Store both numeric labels and English label names.
7. Link rows into leakage groups when they share a source segment, exact English text, or exact Turkish text.
8. Validate required fields and write the aligned TSV safely.

`scripts/split_xed_tr.py` then assigns whole leakage groups to TRAIN, DEV, or TEST. It uses seed 42, exact row capacities, label-aware greedy assignment, and deterministic equal-size group swaps to improve label balance.

The independent `scripts/validate_data.py` does not trust those scripts blindly. It reconstructs the raw pairing and metadata labels again, checks full-row fidelity, and verifies every split and manifest relation.

## Why is leakage dangerous?

Leakage occurs when information from evaluation data is available during development or training. With subtitles, exact duplicates and aligned language counterparts are especially dangerous. If “I am afraid” occurs in TRAIN and the identical line occurs in TEST, the model may succeed by memorization. If English is in TRAIN and its Turkish counterpart is in TEST, cross-lingual performance is also contaminated.

The fixed split prevents an alignment ID, leakage group, exact English text, exact Turkish text, bilingual text pair, or English source segment from crossing partitions. All observed overlap counts are zero. The protection does not catch semantic paraphrases, and it deliberately does not group entire movies, so those remain limitations.

## Why are TRAIN, DEV, and TEST separate?

TRAIN is the only partition used to fit LoRA parameters and select the five demonstrations. DEV is used for prompt wording, decoding choices, PEFT hyperparameters, and checkpoint selection. TEST is reserved for one final evaluation after the final settings are locked.

The proportions are approximately 70/20/10:

- TRAIN: 4,852 rows (70.0043%);
- DEV: 1,386 rows (19.9971%);
- TEST: 693 rows (9.9986%).

The slight decimal differences arise because 6,931 rows cannot be divided into exact integer counts at all three percentages.

## What does zero-shot mean?

Zero-shot prompting gives the model the task instruction and the subtitle but no labelled examples. It measures how well the pretrained model follows the definition directly. Both Qwen and Gemma are evaluated zero-shot in English and Turkish.

The instruction itself is written in English for both languages. This controls the experiment: changing the instruction language at the same time as the input language would introduce another variable.

## What does five-shot mean?

Five-shot prompting adds exactly five labelled examples before the target subtitle. The examples show the required task and output format without changing model parameters.

The demonstrations are stored in `configs/five_shot_examples.json`. They were selected only from TRAIN with a deterministic rule. Together they cover all eight labels, each has at most two labels, and no selected examples repeat exact English or Turkish text.

## Why do the same five alignment IDs matter?

If English and Turkish used different demonstrations, a performance difference could be caused by example selection rather than language. This project fixes five alignment IDs. English runs show their English versions; Turkish runs show the Turkish counterparts. Qwen and Gemma also use the same demonstrations. This makes the comparison controlled.

## What does prompting measure?

Prompting measures the behavior of the pretrained checkpoint without parameter updates. Zero-shot emphasizes instruction following and existing task knowledge. Five-shot also tests whether a small, fixed set of in-context examples improves label selection or output formatting.

Malformed-output rate is important here. A model may understand the emotion but fail the required JSON format. The project reports that separately instead of hiding it inside classification metrics.

## How is prompt sensitivity checked?

The final wording has not yet been selected. The plan compares two English variants: `definitions_v1`, with fuller structured definitions, and `compact_v1`, with a shorter version of the same task. Both keep all eight labels, the multi-label rule, and the strict JSON-only format.

Using the same Qwen checkpoint and deterministic decoding, each variant will run zero-shot on the full English and Turkish DEV partitions. The variant with the higher mean DEV micro-F1 is selected. The tie-breakers are mean sample Jaccard, lower malformed-output rate, and then variant ID. This four-cell comparison tests wording without a large prompt search. Record the choice and lock it before TEST. It is currently BLOCKED because no local Qwen checkpoint exists.

## Why is a chat-template smoke test required?

The tokenizer for an exact Qwen or Gemma checkpoint determines which conversation roles its chat template accepts. The code therefore does not assume that every Gemma tokenizer supports a `system` role. A tokenizer-only smoke test first tries the standard system instruction for zero-shot and five-shot messages. If the tokenizer rejects it, the test can check a user-preamble format that moves the unchanged English instruction into the first user message. DEV prompting accepts only the format recorded for that checkpoint and chat-template hash. No fallback is used before the local test passes.

## What does PEFT mean?

PEFT means parameter-efficient fine-tuning. Instead of updating every parameter in a large model, PEFT learns a much smaller set of additional parameters. This reduces training memory and produces a compact adapter rather than a second full model copy.

Only Qwen is adapted. Gemma is a prompting-only secondary family, as required by the project design.

## What is LoRA?

LoRA is a PEFT method that adds low-rank trainable updates to linear layers while keeping the base model unchanged. The rank controls the size of those updates. Alpha scales them, and dropout regularizes them. The initial settings are rank 16, alpha 32, and dropout 0.05, with all linear layers targeted for the selected Qwen checkpoint.

The project records the base checkpoint, target modules, rank, alpha, dropout, optimizer, scheduler, learning rate, batch configuration, epochs, maximum length, checkpoint selection, seed, software versions, and runtime device for an actual run.

## What is QLoRA?

QLoRA combines LoRA adapters with a quantized base model to reduce memory further. It can be useful on limited hardware but adds quantization dependencies and hardware constraints. No compatible local model/runtime was available during implementation, so this repository chooses ordinary unquantized LoRA rather than pretending QLoRA was tested. The configuration explicitly records `quantization: none`.

## Why is Qwen the primary model and Gemma secondary?

This division is fixed by the course project brief. Qwen is used for both prompting and adaptation, which allows a direct comparison between in-context learning and learned adapters within one model family. Gemma is used only as a prompting comparison. The repository does not claim that this design proves one family is generally superior.

Exact checkpoint names remain blank until checkpoint directories are placed inside the workspace. For each supplied checkpoint, the project must record and justify the exact identifier, model size, multilingual suitability, licence, local accessibility, and selection rationale. Inventing any of these details would make the reported experiment irreproducible.

## What does each PEFT condition test?

English-only PEFT trains Qwen on English TRAIN. It is evaluated on English to measure in-language adaptation and on Turkish to measure cross-lingual transfer without Turkish training.

Turkish-only PEFT trains on Turkish TRAIN and is evaluated on Turkish. It measures target-language adaptation, subject to the limitation that Turkish targets use projected references.

Bilingual PEFT creates both an English and a Turkish training record for every TRAIN alignment, combines them, and shuffles once with seed 42. It is evaluated in both languages. It tests whether direct exposure to both languages offers a useful balance.

These questions can only be answered after real runs. The current code prepares 4,852 English records, 4,852 Turkish records, and 9,704 bilingual TRAIN records. Corresponding DEV sets contain 1,386, 1,386, and 2,772 records.

## How does output parsing work?

The expected response is a JSON list such as:

```json
["joy", "trust"]
```

The strict parser:

- parses the whole response as JSON;
- requires the top-level value to be a list;
- requires every member to be a string;
- normalizes case and surrounding whitespace;
- accepts only the eight exact English emotion names;
- removes duplicate valid labels and restores fixed label order.

It does not search explanatory prose, translate labels, map “happy” to “joy,” or guess intended meaning. A malformed response produces an empty prediction set and a malformed flag. A literal empty JSON list is well formed but predicts no labels.

## What do the metrics mean?

### Precision

Precision asks: of the labels predicted positive, how many were in the reference set? High precision means few extra labels were predicted.

### Recall

Recall asks: of the reference labels, how many were predicted? High recall means few required labels were missed.

### Micro-F1

Micro-F1 pools true positives, false positives, and false negatives over all examples and labels before calculating F1. Frequent labels have more influence. It is a primary metric because it summarizes overall decision quality across the complete task.

### Macro-F1

Macro-F1 computes F1 separately for each emotion and averages the eight scores equally. Rare emotions therefore count as much as frequent ones. A model can have reasonable micro-F1 but weak macro-F1 if it mainly performs well on common labels.

### Sample-averaged Jaccard

For each subtitle, Jaccard divides the size of the intersection between predicted and reference label sets by the size of their union. The score is then averaged across subtitles. It directly measures set overlap and gives partial credit for a multi-label prediction that is partly correct. It is the second primary metric.

### Hamming loss

Hamming loss is the proportion of the eight binary label decisions that are wrong. Lower is better. Predicting one extra label and missing one reference label produces two wrong decisions.

### Exact match

Exact match is the proportion of subtitles for which the predicted set is identical to the reference set. It is strict: a three-label reference with two correct predicted labels still counts as a failure.

### Per-label metrics and support

Per-emotion precision, recall, and F1 show which labels are easy or difficult. Support is the number of reference examples containing that emotion. Support helps prevent overinterpreting an unstable score from a rare label.

### Malformed-output rate

This is the fraction of prompting outputs that violate the JSON-label contract. It distinguishes formatting failures from ordinary wrong label choices.

All undefined precision/recall/F1 cases use the same zero-division value: 0.0.

## How should final tables be interpreted?

Start with micro-F1 and sample Jaccard because they are primary. Then check macro-F1 and per-label results to see whether performance is broad or dominated by frequent emotions. Compare English and Turkish only within the same model and condition. Compare zero-shot and five-shot within the same family. Compare Qwen prompting and Qwen PEFT without treating Gemma as fine-tuned, because it is not.

For cross-lingual transfer, the key row is English-only PEFT evaluated on Turkish. Turkish-only and bilingual results help show whether direct Turkish exposure changes that outcome. Any difference should be described as an observed experimental difference, not automatically as a cultural explanation.

Malformed rate should be read beside classification scores. A system that often violates the output contract is less reliable even if its valid responses are strong.

## What conclusions would be justified?

After real runs, justified claims can be narrow and comparative, for example:

- one condition achieved higher micro-F1 on this fixed TEST set;
- five-shot prompting reduced malformed responses for one checkpoint;
- English-only adaptation transferred better or worse to the paired Turkish inputs than zero-shot prompting;
- one emotion had low recall under several conditions.

Every such claim must point to saved predictions and metric files.

## What conclusions are not justified?

The project cannot conclude that:

- Turkish speakers express emotions in a fundamentally different way;
- the Turkish labels are independent human judgments;
- one model family is universally better than another;
- a difference is cultural rather than caused by translation, projection, subtitle context, checkpoint choice, or sampling;
- a blocked experiment succeeded;
- a metric exists when no prediction file exists.

## How to explain this project to the instructor

A concise explanation is:

> This is a controlled English/Turkish multi-label emotion experiment on XED. Each aligned subtitle pair can have several of eight emotions. Labels come from English source metadata, and Turkish labels are treated as projected references. The project uses one group-aware 70/20/10 split with seed 42. Alignment IDs, source segments, leakage groups, and exact texts do not cross partitions. Qwen is used for prompting and LoRA; Gemma is prompting-only. Zero-shot and five-shot runs use the same English instruction, and five-shot runs use the same five TRAIN alignment IDs in both languages. Outputs must be strict JSON. The primary metrics are micro-F1 and sample Jaccard. All choices are made on DEV, and the final settings must be locked before TEST. Model experiments remain blocked because no local checkpoint is available.

If asked why the data split is trustworthy, mention the independent validator and zero observed overlaps. If asked why Turkish results need caution, explain projected labels and subtitle alignment. If asked why there are no model scores, explain the local-only rule and show `results/master_results.csv`, where all metric cells remain empty. This reports the current project state accurately.

