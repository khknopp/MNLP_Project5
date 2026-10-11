"""Run with: python -m unittest tests.test_helper_training -v

The sampler checks run anywhere; the mixture checks read the real corpus, so they are skipped until
`data/raw/` has been extracted.
"""

import unittest

from src.data import TedData, align
from src.helper_training import RUNS, TARGET_CAPS, MixtureBatchSampler, build_mixture, system_name

HAVE_RAW = all((align.RAW / s / f"{l}.txt").exists() for s in align.SPLIT_DIR for l in align.PROJECT_LANGS)


class MixtureBatchSamplerTests(unittest.TestCase):
    groups = {"az": list(range(10)), "be": list(range(7)), "uk": list(range(100))}

    def owner(self, i):
        # global index -> group, given the dict-order concatenation 10 + 7 + 100
        return "az" if i < 10 else "be" if i < 17 else "uk"

    def test_one_group_per_batch_and_indices_in_range(self):
        for batch in MixtureBatchSampler(self.groups, alpha=0.5, batch_size=4, steps=200):
            self.assertEqual(len({self.owner(i) for i in batch}), 1)
            self.assertTrue(all(0 <= i < 117 for i in batch))

    def test_length_is_steps(self):
        s = MixtureBatchSampler(self.groups, alpha=0.5, batch_size=4, steps=37)
        self.assertEqual(len(s), 37)
        self.assertEqual(len(list(s)), 37)

    def test_same_seed_same_batches_and_new_pass_new_order(self):
        a = MixtureBatchSampler(self.groups, alpha=0.5, batch_size=4, steps=50, seed=1)
        b = MixtureBatchSampler(self.groups, alpha=0.5, batch_size=4, steps=50, seed=1)
        first = list(a)
        self.assertEqual(first, list(b))
        self.assertNotEqual([self.owner(x[0]) for x in first], [self.owner(x[0]) for x in a])

    def test_low_alpha_upsamples_small_groups(self):
        def share(alpha):
            batches = list(MixtureBatchSampler(self.groups, alpha=alpha, batch_size=2, steps=4000))
            return sum(self.owner(b[0]) == "uk" for b in batches) / len(batches)
        self.assertGreater(share(1.0), 0.75)  # natural mixture: 100 / 117
        self.assertLess(share(0.2), 0.5)


@unittest.skipUnless(HAVE_RAW, "data/raw/ not extracted")
class BuildMixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ted = TedData(download=False)
        cls.mix = {run: build_mixture(cls.ted, run) for run in RUNS}

    def test_target_groups_match_the_baseline_training_data(self):
        baseline = self.ted.examples("train", ["az-en", "en-az", "be-en", "en-be"],
                                     {"az": "azj_Latn", "be": "bel_Cyrl", "en": "eng_Latn"})
        for run, groups in self.mix.items():
            if run in TARGET_CAPS:  # these cut a target on purpose: test_az1k_... covers them
                continue
            with self.subTest(run=run):
                self.assertEqual(groups["az"] + groups["be"], baseline)

    def test_az1k_runs_share_one_az_subset_per_seed(self):
        none, tr = build_mixture(self.ted, "none_az1k", seed=43), build_mixture(self.ted, "tr_az1k", seed=43)
        self.assertEqual(len(none["az"]), 2000)
        self.assertEqual(none["az"], tr["az"])
        self.assertEqual(none["be"], self.mix["none"]["be"])
        self.assertNotEqual(none["az"], build_mixture(self.ted, "none_az1k", seed=44)["az"])
        full = {(r["src"], r["tgt"]) for r in self.mix["none"]["az"]}
        self.assertTrue(all((r["src"], r["tgt"]) in full for r in none["az"]))

    def test_none_has_no_helper(self):
        self.assertEqual(list(self.mix["none"]), ["az", "be"])

    def test_system_names_carry_the_seed(self):
        self.assertEqual(system_name("ru_matched", 43), "nllb_helper_ru_matched_s43")

    def test_helper_sizes(self):
        uk, ru = len(self.ted.pairs("train", "uk", "en")), len(self.ted.pairs("train", "ru", "en"))
        self.assertEqual(len(self.mix["uk"]["uk"]), 2 * uk)
        self.assertEqual(len(self.mix["ru_matched"]["ru"]), 2 * uk)
        self.assertEqual(len(self.mix["ru"]["ru"]), 2 * ru)
        self.assertEqual(len(self.mix["tr"]["tr"]), 2 * len(self.ted.pairs("train", "tr", "en")))

    def test_matched_subset_is_a_seeded_subset_of_full_russian(self):
        full = {(r["src"], r["tgt"]) for r in self.mix["ru"]["ru"]}
        sub = self.mix["ru_matched"]["ru"]
        self.assertTrue(all((r["src"], r["tgt"]) in full for r in sub))
        self.assertEqual(sub, build_mixture(self.ted, "ru_matched")["ru"])
        self.assertNotEqual(sub, build_mixture(self.ted, "ru_matched", seed=7)["ru"])

    def test_both_directions_of_each_helper_pair(self):
        rows = self.mix["ru_matched"]["ru"]
        for fwd, bwd in zip(rows[::2], rows[1::2]):
            self.assertEqual((fwd["direction"], bwd["direction"]), ("ru-en", "en-ru"))
            self.assertEqual((fwd["src"], fwd["tgt"]), (bwd["tgt"], bwd["src"]))


if __name__ == "__main__":
    unittest.main()
