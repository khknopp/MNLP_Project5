"""Run with: python -m unittest discover -s tests -v

These read the real corpus, so they are skipped until `data/raw/` has been extracted
(`make data`, or `TedData()` once); the segmentation checks also need `make vocabs`.
"""

import tempfile
import unittest
from pathlib import Path

from src.data import TedData, align

HAVE_RAW = all((align.RAW / s / f"{l}.txt").exists() for s in align.SPLIT_DIR for l in align.PROJECT_LANGS)
VOCABS = align.ROOT / "artifacts" / "vocabularies"
PARALLEL = align.ROOT / "data" / "processed" / "parallel"


@unittest.skipUnless(HAVE_RAW, "data/raw/ not extracted")
class TedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ted = TedData(download=False)

    def test_examples_reproduce_the_nllb_notebook(self):
        # baselines_nllb.ipynb's build(), verbatim except that it reads the raw lines through align.py
        codes = {"az": "azj_Latn", "be": "bel_Cyrl", "en": "eng_Latn"}
        for split in ["train", "dev", "test"]:
            raw = {lang: align._raw(split, lang) for lang in codes}
            expected = []
            for low in ["az", "be"]:
                for x, e in zip(raw[low], raw["en"]):
                    x, e = x.strip(), e.strip()
                    if not (x and e):
                        continue
                    expected.append({"src": x, "tgt": e, "src_lang": codes[low],
                                     "tgt_lang": codes["en"], "direction": f"{low}-en"})
                    expected.append({"src": e, "tgt": x, "src_lang": codes["en"],
                                     "tgt_lang": codes[low], "direction": f"en-{low}"})
            with self.subTest(split=split):
                self.assertEqual(self.ted.examples(split, ["az-en", "en-az", "be-en", "en-be"], codes), expected)

    def test_reverse_direction_swaps_the_sides(self):
        forward = self.ted.pairs("dev", "az", "en")
        self.assertEqual(self.ted.pairs("dev", "en", "az"), [(p, e, a) for p, a, e in forward])

    def test_pairs_match_the_delivered_parallel_files(self):
        if not (PARALLEL / "az-en.test.pos").exists():
            self.skipTest("data/processed/parallel/ not emitted")
        positions, az, en = zip(*self.ted.pairs("test", "az", "en"))
        read = lambda name: (PARALLEL / name).read_text(encoding="utf-8").splitlines()
        self.assertEqual([str(p) for p in positions], read("az-en.test.pos"))
        self.assertEqual(list(az), read("az-en.test.az"))
        self.assertEqual(list(en), read("az-en.test.en"))

    def test_check_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write = lambda name, lines: (folder / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            write("en-az.ref", self.ted.references("en-az"))
            write("az-en.ref", self.ted.references("az-en")[:100])
            write("en-be.ref", ["not the reference"] + self.ted.references("en-be")[1:])
            write("xx-en.ref", ["x"])
            self.assertEqual(self.ted.check_references(folder), {
                "az-en.ref": f"OK: first 100 of {len(self.ted.references('az-en'))}",
                "en-az.ref": "OK",
                "en-be.ref": "DIFF",
                "xx-en.ref": "unknown direction",
            })

    def test_unknown_split_or_language_is_rejected(self):
        for args in [("validation", "az", "en"), ("test", "xx", "en"), ("test", "en", "en")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.ted.pairs(*args)

    def test_v5_maps_characters_before_segmenting(self):
        if not (VOCABS / "V5_8000_unigram" / "tokenizer.json").exists():
            self.skipTest("vocabularies not trained")
        mapped, plain = "gözlənilməz xəbər", "gözlenilmez heber"
        self.assertEqual(self.ted.segment([mapped], "az", "V5_8000_unigram"),
                         self.ted.segment([plain], "az", "V5_8000_unigram"))
        self.assertNotEqual(self.ted.segment([mapped], "az", "V3_8000_unigram"),
                            self.ted.segment([plain], "az", "V3_8000_unigram"))


if __name__ == "__main__":
    unittest.main()
