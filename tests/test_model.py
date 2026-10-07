"""Run with: python -m unittest discover -s tests -v"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src import evaluation
from src.model import MODELS, CopyModel, NLLBModel, Seq2SeqModel, Vocabulary, build_model, save_translations, translate_corpus

LOCAL_MODELS = {"seq2seq": Seq2SeqModel, "copy": CopyModel}


class VocabularyTests(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        vocab = Vocabulary(["hello world", "hello there"])
        self.assertEqual(vocab.decode_sentence(vocab.encode_sentence("hello world")), "hello world")

    def test_unknown_token_maps_to_unk(self):
        vocab = Vocabulary(["hello world"])
        self.assertEqual(vocab.decode_sentence(vocab.encode_sentence("goodbye")), "<UNK>")


class CopyModelTests(unittest.TestCase):
    def setUp(self):
        self.model = CopyModel(Vocabulary(["a b"]), Vocabulary(["c d"]))

    def test_fit_accepts_and_ignores_any_kwargs(self):
        self.model.fit(pairs=[("a", "b")], epochs=5, learning_rate=0.1)

    def test_predict_returns_the_source_sentence(self):
        self.assertEqual(self.model.predict("a b", max_length=1, verbose=False), "a b")


class RegistryTests(unittest.TestCase):
    def test_build_model_constructs_the_registered_class(self):
        for name, cls in LOCAL_MODELS.items():
            with self.subTest(name=name):
                model = build_model(name, Vocabulary(["a"]), Vocabulary(["b"]))
                self.assertIsInstance(model, cls)

    def test_seq2seq_is_registered(self):
        self.assertIs(MODELS["seq2seq"], Seq2SeqModel)

    def test_nllb_is_registered(self):
        self.assertIs(MODELS["nllb"], NLLBModel)

    def test_unknown_model_name_raises(self):
        with self.assertRaises(KeyError):
            build_model("does-not-exist", Vocabulary(["a"]), Vocabulary(["b"]))


class NLLBModelTests(unittest.TestCase):
    def build(self, **kwargs):
        tokenizer = MagicMock()
        tokenizer.convert_tokens_to_ids.return_value = 7
        tokenizer.batch_decode.return_value = ["salam dunya"]
        tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        hf_model = MagicMock()
        hf_model.generate.return_value = [[0, 1, 2]]
        patcher = patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer)
        self.tok_load = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=hf_model)
        self.model_load = patcher.start()
        self.addCleanup(patcher.stop)
        return build_model("nllb", None, None, **kwargs), tokenizer, hf_model

    def test_language_codes_are_resolved(self):
        model, *_ = self.build(src_lang="en", tgt_lang="az")
        self.assertEqual((model.src_lang, model.tgt_lang), ("eng_Latn", "azj_Latn"))

    def test_unmapped_language_is_passed_through(self):
        model, *_ = self.build(src_lang="eng_Latn", tgt_lang="xx_Yyyy")
        self.assertEqual((model.src_lang, model.tgt_lang), ("eng_Latn", "xx_Yyyy"))

    def test_fit_is_a_noop(self):
        model, *_ = self.build()
        model.fit(pairs=[("a", "b")], epochs=5)

    def test_predict_decodes_the_generated_ids(self):
        model, tokenizer, hf_model = self.build(src_lang="en", tgt_lang="az")
        self.assertEqual(model.predict("hello world", max_length=5, verbose=False), "salam dunya")
        self.assertEqual(tokenizer.src_lang, "eng_Latn")
        hf_model.generate.assert_called_once()
        self.assertEqual(hf_model.generate.call_args.kwargs["forced_bos_token_id"], 7)
        self.assertEqual(hf_model.generate.call_args.kwargs["max_new_tokens"], 5)

    def test_no_network_call_happens(self):
        self.build()
        self.tok_load.assert_called_once()
        self.model_load.assert_called_once()


class CorpusHelpersTests(unittest.TestCase):
    def test_translate_corpus_preserves_order_and_length(self):
        model = CopyModel(Vocabulary(["a b c"]), Vocabulary(["a b c"]))
        sources = ["a b", "b c", "c"]
        self.assertEqual(translate_corpus(model, sources), sources)

    def test_save_translations_writes_the_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_translations(out, "en-az", ["s1", "s2"], ["r1", "r2"], ["h1", "h2"])
            self.assertEqual((out / "en-az.src").read_text(), "s1\ns2\n")
            self.assertEqual((out / "en-az.ref").read_text(), "r1\nr2\n")
            self.assertEqual((out / "en-az.hyp").read_text(), "h1\nh2\n")

    def test_copy_model_scores_perfectly_when_source_equals_reference(self):
        model = CopyModel(Vocabulary(["a b c d"]), Vocabulary(["a b c d"]))
        pairs = [("a b c d", "a b c d"), ("e f g h", "e f g h")]
        hyps = translate_corpus(model, [s for s, _ in pairs])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_translations(out, "xx-yy", [s for s, _ in pairs], [t for _, t in pairs], hyps)
            result = evaluation.evaluate_direction(out, "xx-yy", "bleu")
        self.assertAlmostEqual(result["score"], 100)


if __name__ == "__main__":
    unittest.main()


class AttentionSeq2SeqTests(unittest.TestCase):
    def test_learns_a_tiny_corpus_and_translates(self):
        import random

        import torch

        from src.model import AttentionSeq2Seq
        random.seed(0)
        torch.manual_seed(0)
        pairs = [("a b c", "x y z"), ("c b a", "z y x"), ("a a b", "x x y"), ("b c c", "y z z")] * 8
        model = AttentionSeq2Seq.from_pairs(pairs, vocab_size=20, embedding_dim=32, hidden_dim=32, dropout=0.0)
        model.fit(pairs, dev_pairs=pairs[:4], epochs=30, learning_rate=5e-3, batch_size=8, verbose=False)
        self.assertEqual(model.predict("a b c", max_length=10, verbose=False), "x y z")
