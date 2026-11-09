from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from xed_project.constants import LABELS, PARSER_VERSION, PROMPT_VERSION
from xed_project.chat_templates import adapt_messages_for_chat_template
from xed_project.metrics import evaluate_label_sets
from xed_project.parsing import parse_prediction
from xed_project.prompting import (
    DEFAULT_PROMPT_VARIANT,
    PROMPT_INSTRUCTIONS,
    Demonstration,
    INSTRUCTION,
    build_messages,
)
from select_prompt_variant import rank_summaries


class ParserTests(unittest.TestCase):
    def test_valid_case_whitespace_and_duplicates(self) -> None:
        result = parse_prediction('[" Joy ", "trust", "JOY"]')
        self.assertEqual(result.labels, ("joy", "trust"))
        self.assertFalse(result.malformed)
        self.assertEqual(result.parser_version, PARSER_VERSION)

    def test_explanatory_prose_is_not_recovered(self) -> None:
        result = parse_prediction('The answer is ["joy"]')
        self.assertEqual(result.labels, ())
        self.assertTrue(result.malformed)

    def test_invalid_member_invalidates_whole_list(self) -> None:
        result = parse_prediction('["joy", "happiness"]')
        self.assertEqual(result.labels, ())
        self.assertTrue(result.malformed)

    def test_empty_list_is_well_formed_empty_prediction(self) -> None:
        result = parse_prediction("[]")
        self.assertEqual(result.labels, ())
        self.assertFalse(result.malformed)


class PromptTests(unittest.TestCase):
    def test_same_instruction_for_both_languages(self) -> None:
        english = build_messages("Hello.", "en")
        turkish = build_messages("Merhaba.", "tr")
        self.assertEqual(english[0], turkish[0])
        self.assertEqual(english[0]["content"], INSTRUCTION)
        self.assertIn("one or more", INSTRUCTION)
        for label in LABELS:
            self.assertIn(label, INSTRUCTION)

    def test_both_prompt_variants_preserve_the_output_contract(self) -> None:
        self.assertEqual(set(PROMPT_INSTRUCTIONS), {"definitions_v1", "compact_v1"})
        for variant, instruction in PROMPT_INSTRUCTIONS.items():
            self.assertIn("one or more", instruction)
            self.assertIn("JSON", instruction)
            for label in LABELS:
                self.assertIn(label, instruction, variant)
            english = build_messages("Hello.", "en", prompt_variant=variant)
            turkish = build_messages("Merhaba.", "tr", prompt_variant=variant)
            self.assertEqual(english[0], turkish[0])

    def test_chat_template_compatibility_modes_preserve_instruction(self) -> None:
        messages = build_messages("Target.", "en")
        system = adapt_messages_for_chat_template(messages, "system")
        self.assertEqual(system, messages)
        preamble = adapt_messages_for_chat_template(messages, "user_preamble")
        self.assertNotEqual(preamble[0]["role"], "system")
        self.assertIn(PROMPT_INSTRUCTIONS[DEFAULT_PROMPT_VARIANT], preamble[0]["content"])
        self.assertIn("Target.", preamble[0]["content"])

    def test_prompt_variant_ranking_uses_preregistered_tie_breaks(self) -> None:
        summaries = [
            {
                "variant": "definitions_v1",
                "mean_micro_f1": 0.5,
                "mean_sample_jaccard": 0.4,
                "mean_malformed_output_rate": 0.02,
            },
            {
                "variant": "compact_v1",
                "mean_micro_f1": 0.5,
                "mean_sample_jaccard": 0.4,
                "mean_malformed_output_rate": 0.01,
            },
        ]
        ranked = rank_summaries(summaries)
        self.assertEqual(ranked[0]["variant"], "compact_v1")

    def test_five_shot_requires_exactly_five(self) -> None:
        example = Demonstration("id", "text", ("joy",))
        with self.assertRaises(ValueError):
            build_messages("target", "en", [example])
        messages = build_messages("target", "en", [example] * 5)
        self.assertEqual(len(messages), 12)


class MetricTests(unittest.TestCase):
    def test_known_multilabel_case(self) -> None:
        result = evaluate_label_sets(
            [("anger", "joy"), ("trust",)],
            [("anger",), ("trust", "surprise")],
            [False, True],
        )
        self.assertAlmostEqual(result.overall["micro_precision"], 2 / 3)
        self.assertAlmostEqual(result.overall["micro_recall"], 2 / 3)
        self.assertAlmostEqual(result.overall["micro_f1"], 2 / 3)
        self.assertAlmostEqual(result.overall["sample_jaccard"], 0.5)
        self.assertAlmostEqual(result.overall["hamming_loss"], 2 / 16)
        self.assertEqual(result.overall["exact_match_ratio"], 0.0)
        self.assertEqual(result.overall["malformed_output_rate"], 0.5)


class ArtifactTests(unittest.TestCase):
    def test_five_shot_examples_are_exactly_train_only(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "five_shot_examples.json").read_text(encoding="utf-8")
        )
        examples = config["examples"]
        self.assertEqual(config["source_split"], "train")
        self.assertEqual(len(examples), 5)
        ids = [example["alignment_id"] for example in examples]
        self.assertEqual(len(ids), len(set(ids)))
        covered = {label for example in examples for label in example["gold_labels"]}
        self.assertEqual(covered, set(LABELS))

        def ids_in(path: Path) -> set[str]:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return {row["alignment_id"] for row in csv.DictReader(handle, delimiter="\t")}

        train_ids = ids_in(ROOT / "data" / "splits" / "train.tsv")
        dev_ids = ids_in(ROOT / "data" / "splits" / "dev.tsv")
        test_ids = ids_in(ROOT / "data" / "splits" / "test.tsv")
        self.assertTrue(set(ids) <= train_ids)
        self.assertFalse(set(ids) & dev_ids)
        self.assertFalse(set(ids) & test_ids)

    def test_final_config_remains_unfrozen_without_models(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "final_experiment_config.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["status"], "DRAFT_NOT_FROZEN")
        self.assertEqual(config["prompt_version"], PROMPT_VERSION)
        self.assertIsNone(config["prompt_variant"])
        self.assertIsNone(config["prompt_selection_sha256"])
        self.assertEqual(config["parser"]["version"], PARSER_VERSION)

    def test_prompt_sensitivity_plan_is_small_and_dev_only(self) -> None:
        plan = json.loads(
            (ROOT / "configs" / "prompt_sensitivity.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(plan["split"], "dev")
        self.assertEqual(plan["model_family"], "qwen")
        self.assertEqual(plan["condition"], "zero_shot")
        self.assertEqual(plan["languages"], ["en", "tr"])
        self.assertEqual(
            {variant["id"] for variant in plan["variants"]},
            set(PROMPT_INSTRUCTIONS),
        )
        self.assertNotIn("test", json.dumps(plan["controls"]).lower())
        prompt_config = json.loads(
            (ROOT / "configs" / "prompting.json").read_text(encoding="utf-8")
        )
        self.assertEqual(prompt_config["status"], "DEV_SELECTED")
        self.assertEqual(prompt_config["selected_variant"], "definitions_v1")
        self.assertEqual(
            prompt_config["selection_artifact"],
            "results/prompting/prompt_sensitivity/selection.json",
        )

    def test_local_model_artifacts_are_ignored(self) -> None:
        rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for required in (
            "configs/models.local.json",
            "configs/chat_templates.local.json",
            "results/peft/checkpoints/",
            "__pycache__/",
            ".DS_Store",
            "*.partial",
        ):
            self.assertIn(required, rules)

    def test_completed_results_have_no_metrics(self) -> None:
        with (ROOT / "results" / "master_results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 13)
        metric_fields = (
            "micro_f1",
            "sample_jaccard",
            "macro_f1",
            "micro_precision",
            "micro_recall",
            "hamming_loss",
            "exact_match_ratio",
            "malformed_output_rate",
        )
        for row in rows:
            self.assertEqual(row["status"], "COMPLETED")
            self.assertFalse(row["blocker"])
            self.assertTrue(all(row[field] != "" for field in metric_fields))


if __name__ == "__main__":
    unittest.main()
