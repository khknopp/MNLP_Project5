#!/usr/bin/env python3
"""Are the RQ1 differences real? Paired bootstrap over test sentences, on scores averaged over seeds.

    python scripts/significance_rq1.py                              # reads results/
    python scripts/significance_rq1.py --results results_main_h100 --out results_main_h100/significance.md

COMET is included when every compared run has a results/metrics/<system>/comet.json
(scripts/comet_rq1.py); COMET is the mean of sentence scores, shown x100.

For each comparison A vs B and each direction: resample the test sentences with replacement (the same
sample for every system and seed, so the test is paired), score every seed's translations on that
sample, average over seeds, and take A - B. Repeated `--samples` times this gives:

    delta   observed difference of the seed averages (BLEU / chrF points)
    95% CI  2.5th-97.5th percentile of the resampled differences
    p       two-sided: how often the resampled difference falls on the other side of zero (x2)
    seeds   in how many seeds A beat B, pairing equal seed numbers

The bootstrap covers test-set variance; `seeds` shows whether training randomness agrees. Call a
difference real only when p < 0.05 AND the seeds mostly agree.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ["az-en", "en-az", "be-en", "en-be"]
COMPARISONS = [  # (A, B, what it answers)
    ("uk", "none", "does Ukrainian help?"),
    ("ru_matched", "none", "does Russian (108k) help?"),
    ("ru", "none", "does Russian (208k) help?"),
    ("tr", "none", "does Turkish help?"),
    ("uk", "ru_matched", "similarity: uk vs ru at equal size"),
    ("ru", "ru_matched", "volume: ru 208k vs 108k"),
    ("ru", "uk", "as asked: all Russian vs all Ukrainian"),
    ("tr", "uk", "related vs unrelated: az: tr related; be: uk related"),
    ("tr", "ru", "related vs unrelated, similar size (182k vs 208k)"),
    ("tr_az1k", "none_az1k", "az cut to 1k pairs: does Turkish help when az is scarce?"),
    ("none_az1k", "none", "how much does cutting az to 1k cost?"),
    ("tr_scratch", "none_scratch", "FROM SCRATCH: does Turkish help az without pretraining?"),
    ("uk_scratch", "none_scratch", "FROM SCRATCH: does Ukrainian help?"),
    ("ru_matched_scratch", "none_scratch", "FROM SCRATCH: does Russian (108k) help?"),
    ("ru_scratch", "none_scratch", "FROM SCRATCH: does Russian (208k) help?"),
    ("uk_scratch", "ru_matched_scratch", "FROM SCRATCH: similarity: uk vs ru at equal size"),
    ("ru_scratch", "ru_matched_scratch", "FROM SCRATCH: volume: ru 208k vs 108k"),
    ("ru_scratch", "uk_scratch", "FROM SCRATCH: as asked: all Russian vs all Ukrainian"),
    ("tr_scratch", "ru_scratch", "FROM SCRATCH: related vs unrelated, similar size (az: tr related; be: ru related)"),
    ("tr_scratch", "uk_scratch", "FROM SCRATCH: related vs unrelated (az: tr related; be: uk related)"),
]
# sacrebleu's own sufficient statistics: scoring summed per-sentence stats reproduces corpus_score exactly
METRICS = {"BLEU": BLEU(), "chrF": CHRF()}


def load(results: Path) -> dict[str, dict[int, Path]]:
    runs: dict[str, dict[int, Path]] = {}
    for d in sorted((results / "translations").iterdir()):
        m = re.fullmatch(r"nllb_helper_(.+?)_s(\d+)(_scratch)?", d.name)
        if m:
            runs.setdefault(m.group(1) + (m.group(3) or ""), {})[int(m.group(2))] = d
    return runs


def lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, help="also write the tables to this markdown file")
    args = ap.parse_args()

    runs = load(args.results)
    comparisons = [c for c in COMPARISONS if c[0] in runs and c[1] in runs]
    for a, b, _ in COMPARISONS:
        if a not in runs or b not in runs:
            print(f"(skipped {a} vs {b}: not on disk under {args.results})")
    needed = {r for a, b, _ in comparisons for r in (a, b)}
    seeds = sorted(set.intersection(*(set(runs[r]) for r in needed)))
    comet_file = lambda folder: args.results / "metrics" / folder.name / "comet.json"
    names = list(METRICS) + (["COMET"] if all(comet_file(runs[r][s]).exists() for r in needed for s in seeds) else [])
    comet = {}

    rng = np.random.default_rng(args.seed)
    out = [f"Paired bootstrap, {args.samples} samples, seeds {seeds} (scores averaged over seeds)\n"]
    for d in DIRECTIONS:
        refs = lines(runs["none"][seeds[0]] / f"{d}.ref")
        n = len(refs)
        # counts[i, j] = how often sentence j is in bootstrap sample i; the same samples for every system
        counts = rng.multinomial(n, np.full(n, 1 / n), size=args.samples).astype(np.float64)
        observed, resampled = {}, {}
        for run in needed:
            for metric in names:
                obs, boot = [], []
                for s in seeds:
                    folder = runs[run][s]
                    if lines(folder / f"{d}.ref") != refs:
                        raise SystemExit(f"{folder}/{d}.ref differs: systems were not scored on the same test set")
                    if metric == "COMET":  # system score = mean of sentence scores
                        if folder not in comet:
                            comet[folder] = json.loads(comet_file(folder).read_text())
                        st = 100 * np.asarray(comet[folder][d]["sentence_scores"], dtype=np.float64)
                        obs.append(st.mean())
                        boot.append(counts @ st / n)
                        continue
                    M = METRICS[metric]
                    st = np.asarray(M._extract_corpus_statistics(lines(folder / f"{d}.hyp"), [refs]), dtype=np.float64)
                    obs.append(M._compute_score_from_stats(list(st.sum(0))).score)
                    sums = counts @ st
                    boot.append([M._compute_score_from_stats(list(row)).score for row in sums])
                observed[run, metric] = np.array(obs)
                resampled[run, metric] = np.mean(boot, axis=0)

        out.append(f"\n### {d}\n")
        out.append("| comparison | question | " + " | ".join(f"{m} delta [95% CI] | p | seeds" for m in names) + " |")
        out.append("|---|---|" + "---|---|---|" * len(names))
        for a, b, question in comparisons:
            cells = []
            for metric in names:
                delta = observed[a, metric].mean() - observed[b, metric].mean()
                diff = resampled[a, metric] - resampled[b, metric]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
                wins = int((observed[a, metric] > observed[b, metric]).sum())
                star = " **" if p < 0.05 else ""
                cells += [f"{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]", f"{p:.3f}{star}", f"{wins}/{len(seeds)}"]
            out.append(f"| {a} vs {b} | {question} | " + " | ".join(cells) + " |")
    out.append("\n** = p < 0.05. seeds = in how many seeds A scored higher than B.")
    text = "\n".join(out)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
