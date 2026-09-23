"""Run with: python -m unittest discover -s tests -v (requires sacrebleu).

COMET is simulated here: these tests do not download a neural model.
Saved-score checks are skipped when the local baseline artifacts are absent.
"""

import builtins
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import evaluation as evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for direction in evaluation.DIRECTIONS:
            for extension in ("src", "hyp", "ref"):
                self.write(direction, extension, "This is a complete sentence.\nAnother complete sentence is here.\n")

    def write(self, direction, extension, text):
        (self.folder / f"{direction}.{extension}").write_bytes(text.encode("utf-8"))

    def test_invalid_metric_fails_before_io_or_loading(self):
        with patch.object(evaluation, "_translation_paths") as paths, patch.object(evaluation, "load_metric") as loader:
            for name in ("", "accuracy", "chrF2"):
                for function, args in (
                    (evaluation.evaluate_direction, ("missing", "en-be", name)),
                    (evaluation.evaluate_all_directions, ("missing", name)),
                ):
                    with self.subTest(name=name, function=function.__name__):
                        with self.assertRaisesRegex(ValueError, "Unknown metric"):
                            function(*args)
            paths.assert_not_called()
            loader.assert_not_called()

    def test_empty_directions(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluation.evaluate_all_directions(self.folder, "bleu", directions=[])

    def test_missing_file_in_last_direction_prevents_model_loading(self):
        (self.folder / "en-be.hyp").unlink()
        with patch.object(evaluation, "load_metric") as loader:
            with self.assertRaisesRegex(FileNotFoundError, "en-be.hyp"):
                evaluation.evaluate_all_directions(self.folder, "comet")
            loader.assert_not_called()

    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError):
            evaluation.load_translations(self.folder / "missing", "en-be")

    def test_mismatched_counts_prevent_model_loading(self):
        for extension in ("src", "ref", "hyp"):
            with self.subTest(extension=extension):
                path = self.folder / f"en-be.{extension}"
                original = path.read_bytes()
                self.write("en-be", extension, "one line\n")
                with patch.object(evaluation, "load_metric") as loader:
                    with self.assertRaisesRegex(ValueError, "unequal line counts"):
                        evaluation.evaluate_direction(self.folder, "en-be", "comet")
                    loader.assert_not_called()
                path.write_bytes(original)

    def test_empty_files(self):
        for extension in ("src", "ref", "hyp"):
            self.write("en-be", extension, "")
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluation.load_translations(self.folder, "en-be")

    def test_unicode_blank_lines_and_missing_final_newline(self):
        for extension in ("src", "ref", "hyp"):
            self.write("en-be", extension, "  прывітанне ə  \r\n\r\nlast")
        texts = evaluation.load_translations(self.folder, "en-be")
        self.assertEqual(texts["hyp"], ["  прывітанне ə  ", "", "last"])

    def test_bleu_and_chrf_need_no_source_and_score_perfect_match(self):
        (self.folder / "en-be.src").unlink()
        for name in (" BLEU ", "CHRF"):
            result = evaluation.evaluate_direction(self.folder, "en-be", name)
            self.assertAlmostEqual(result["score"], 100)
            self.assertEqual(result["metric"], name.strip().lower())
            self.assertIn("signature", result)
        with self.assertRaises(FileNotFoundError):
            evaluation.evaluate_direction(self.folder, "en-be", "comet")

    def test_empty_hypotheses_are_scored_not_dropped(self):
        self.write("en-be", "hyp", "\n\n")
        for name in ("bleu", "chrf"):
            result = evaluation.evaluate_direction(self.folder, "en-be", name)
            self.assertEqual(result["score"], 0)
            self.assertEqual(result["n_sentences"], 2)

    def test_custom_chrf_and_keyword_only_settings(self):
        default = evaluation.load_metric("chrf")
        custom = evaluation.load_metric("chrf", beta=1, word_order=2)
        self.assertEqual((default.beta, default.word_order), (2, 0))
        self.assertEqual((custom.beta, custom.word_order), (1, 2))
        result = evaluation.evaluate_direction(self.folder, "en-be", "chrf", metric=custom)
        self.assertIn("nw:2", result["signature"])
        with self.assertRaises(TypeError):
            evaluation.load_metric("chrf", 1)

    def test_comet_format_order_options_and_single_loading(self):
        metric = Mock()
        metric.predict.return_value = SimpleNamespace(system_score=0.6, scores=[0.8, 0.4])
        with patch.object(evaluation, "load_metric", return_value=metric) as loader, patch.object(evaluation, "load_translations", wraps=evaluation.load_translations) as reader:
            results = evaluation.evaluate_all_directions(self.folder, "comet", batch_size=2, gpus=0)
            loader.assert_called_once_with("comet")
            self.assertEqual(reader.call_count, 4)
        self.assertEqual(metric.predict.call_count, 4)
        examples = metric.predict.call_args.args[0]
        self.assertEqual(examples[0], {"src": "This is a complete sentence.", "mt": "This is a complete sentence.", "ref": "This is a complete sentence."})
        self.assertEqual(metric.predict.call_args.kwargs, {"batch_size": 2, "gpus": 0})
        self.assertEqual(results["en-be"]["sentence_scores"], [0.8, 0.4])
        json.dumps(results)
        with patch.object(evaluation, "load_metric") as loader:
            evaluation.evaluate_all_directions(self.folder, "comet", metric=metric)
            loader.assert_not_called()

    def test_relative_paths_from_notebook_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.folder)
            with patch.object(evaluation, "ROOT", self.folder.parent):
                result = evaluation.evaluate_direction(self.folder.name, "en-be", "bleu")
            self.assertAlmostEqual(result["score"], 100)
        finally:
            os.chdir(previous)

    def test_missing_dependencies_raise_import_error(self):
        original_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name.split(".")[0] in ("sacrebleu", "comet"):
                raise ModuleNotFoundError(f"No module named {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            for name in ("bleu", "chrf", "comet"):
                with self.assertRaises(ImportError):
                    evaluation.load_metric(name)

    def test_saved_baseline_scores(self):
        folders = [evaluation.ROOT / "results" / "translations" / name for name in ("nllb_basemodel", "nllb_finetuned_no_helpers")]
        if not all((folder / "en-be.hyp").is_file() for folder in folders):
            self.skipTest("Local baseline artifacts are unavailable")
        for folder in folders:
            metadata = evaluation.ROOT / "results" / "metrics" / folder.name / ("scores.json" if folder.name == "nllb_basemodel" else "run_info.json")
            expected = json.loads(metadata.read_text(encoding="utf-8"))
            if folder.name != "nllb_basemodel":
                expected = expected["test_scores"]
            for name, key in (("bleu", "BLEU"), ("chrf", "chrF")):
                for direction, result in evaluation.evaluate_all_directions(folder, name).items():
                    with self.subTest(system=folder.name, metric=name, direction=direction):
                        self.assertEqual(round(result["score"], 2), expected[direction][key])


if __name__ == "__main__":
    unittest.main()
