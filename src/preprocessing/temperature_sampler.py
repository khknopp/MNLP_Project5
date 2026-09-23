#!/usr/bin/env python3
"""Training-time temperature sampler — required when adding helper languages,
otherwise the experiment's conclusion is not valid.

The project plan says *"temperature-based sampling will be used to rebalance the
training mixture"*. This is it. It decides **how much of each batch comes from each
language** — that is a *data mixture* decision, not a model hyperparameter, which is
why it belongs to WP0 (the preprocessing layer) and not to the training code.

**Why it is not optional**

    ru 208,458 sentences : az 5,946 sentences  ~=  35 : 1

Train on the natural mixture and az/be barely exist in the gradient. Then
"adding helper languages made things worse" gets read as "helper languages don't
help" — when in fact the mixture was never rebalanced. Without this sampler the
conclusion of that experiment is invalid, not merely weak.

**Relation to train_vocabs.py**: same weighting formula (p_i ∝ n_i^alpha), different
consumer — that one mixes the text once for vocabulary training, this one samples per
batch during training. Do not change one without the other.

**Usage**

    # 1) Inspect the mixture first (no training, a few seconds)
    python temperature_sampler.py --alphas 0.2 0.5 0.8 1.0

    # 2) Use it as a batch sampler in training
    from temperature_sampler import TemperatureBatchSampler
    sampler = TemperatureBatchSampler(
        sizes={"az": 5946, "be": 4509, "tr": 182470, "uk": 108495, "ru": 208458, "en": 258098},
        alpha=0.2, batch_size=32, seed=42,
    )
    for lang, idx in sampler:        # each batch comes from exactly one language
        batch = datasets[lang][idx]  # -> fetch the samples yourself

    # 3) Into a PyTorch DataLoader (one Dataset per language)
    loader = DataLoader(datasets[lang], batch_sampler=...)
    # or wrap it in an IterableDataset whose __iter__ does `for lang, idx in sampler`

Rule of thumb: alpha = 1.0 is the natural mixture (i.e. no sampling); alpha = 0.2 is
the setting that recovered the vocabulary slots in this work package (see notebooks/data_overview.ipynb §8, the V4 rows of the cost table). **Do not pick alpha by feel** — run (1) and look at the table.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ core formula
def language_weights(sizes: dict[str, int], alpha: float) -> dict[str, float]:
    """p_i ∝ n_i^alpha, normalised to a probability. alpha = 1 is the natural mixture.

    Same formula as train_vocabs.temperature_mix() — change one, change both.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0; alpha=1 is the natural mixture, alpha<1 upsamples low-resource languages")
    w = {k: float(n) ** alpha for k, n in sizes.items() if n > 0}
    z = sum(w.values())
    if z <= 0:
        raise ValueError("sizes is empty")
    return {k: v / z for k, v in w.items()}


def expected_share(sizes: dict[str, int], alpha: float) -> dict[str, float]:
    """Share of *samples* each language is expected to contribute per epoch.

    Note: p_i is the probability of *picking* a language, but every pick draws a full
    batch_size, so the sample share is proportional to p_i. Both readings appear in the
    report — keep them straight.
    """
    return language_weights(sizes, alpha)


# ------------------------------------------------------------------ sampler
class TemperatureBatchSampler:
    """Pick a language with p_i ∝ n_i^alpha, then draw one batch of indices from it.

    Each iteration yields `(lang, [idx, ...])` — **one language per batch**.
    That is the usual NMT setup: mixing languages inside a batch complicates both
    padding and pair semantics, and mixture control only needs the *sequence of
    languages across batches* to follow p_i.

    To avoid drawing the same samples repeatedly, each language keeps a **shuffled
    index pool** that is reshuffled when exhausted. That preserves the mixture while
    still cycling through each language's own data.
    """

    def __init__(
        self,
        sizes: dict[str, int],
        alpha: float,
        batch_size: int,
        seed: int = 42,
        steps_per_epoch: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.sizes = {k: n for k, n in sizes.items() if n > 0}
        self.alpha = alpha
        self.batch_size = batch_size
        self.seed = seed
        self.weights = language_weights(self.sizes, alpha)
        self.langs = list(self.weights.keys())
        # Default: one epoch = one pass over the natural corpus size, so
        # upsampling does not silently lengthen the epoch.
        self.steps_per_epoch = steps_per_epoch or max(1, sum(self.sizes.values()) // batch_size)
        self._pools: dict[str, list[int]] = {}

    # -- internal: draw batch_size indices for one language (reshuffle when the pool empties)
    def _draw(self, lang: str, rng: random.Random) -> list[int]:
        n = self.sizes[lang]
        pool = self._pools.setdefault(lang, [])
        take = min(self.batch_size, n)
        out: list[int] = []
        while len(out) < take:
            if not pool:
                pool.extend(range(n))
                rng.shuffle(pool)
            out.append(pool.pop())
        return out

    def __iter__(self):
        rng = random.Random(self.seed)
        langs = self.langs
        probs = [self.weights[k] for k in langs]
        for _ in range(self.steps_per_epoch):
            lang = rng.choices(langs, weights=probs, k=1)[0]
            yield lang, self._draw(lang, rng)

    def __len__(self) -> int:
        return self.steps_per_epoch


# ------------------------------------------------------------------ preview table
def _sizes_from_corpus(data_dir: Path, split: str = "train") -> dict[str, int]:
    sizes: dict[str, int] = {}
    for p in sorted(data_dir.glob(f"*.{split}.txt")):
        lang = p.name.split(".")[0]
        n = sum(1 for line in p.open(encoding="utf-8") if line.strip())
        if n:
            sizes[lang] = n
    return sizes


def _parse_sizes(pairs: list[str]) -> dict[str, int]:
    out = {}
    for s in pairs:
        k, _, v = s.partition("=")
        out[k] = int(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sizes", nargs="*", default=None, help="e.g. az=5946 be=4509 ... (defaults to reading the corpus)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.2, 0.5, 0.8, 1.0])
    ap.add_argument("--batch-size", type=int, default=32, help="used to compute steps per epoch")
    args = ap.parse_args()

    sizes = _parse_sizes(args.sizes) if args.sizes else _sizes_from_corpus(args.data_dir, args.split)
    if not sizes:
        raise SystemExit(f"No corpus found at {args.data_dir}/*.{args.split}.txt — run `make data` first")

    total = sum(sizes.values())
    print(f"corpus ({args.split}): " + "  ".join(f"{k}={v:,}" for k, v in sorted(sizes.items(), key=lambda x: -x[1])))
    print(f"total {total:,} sentences   batch_size={args.batch_size}\n")

    langs = sorted(sizes, key=lambda k: -sizes[k])
    print("Natural mixture vs temperature-sampled mixture (by alpha):\n")
    header = f"{'lang':6s}{'corpus':>10s}" + "".join(f"{'a=' + str(a):>11s}" for a in args.alphas)
    print(header)
    print("-" * len(header))
    for lang in langs:
        row = f"{lang:6s}{sizes[lang]:>10,}"
        for a in args.alphas:
            row += f"{expected_share(sizes, a)[lang] * 100:>10.1f}%"
        print(row)
    print()
    for a in args.alphas:
        w = expected_share(sizes, a)
        lo = min(sizes, key=lambda k: sizes[k])
        hi = max(sizes, key=lambda k: sizes[k])
        ratio = w[hi] / max(w[lo], 1e-12)
        steps = max(1, total // args.batch_size)
        print(f"  alpha={a:<4} richest/poorest language share = {ratio:>7.2f} : 1"
              f"   (natural mixture is {sizes[hi] / sizes[lo]:.2f} : 1)"
              f"   one epoch ~= {steps:,} steps")
    print("\nNote: alpha=1.0 means no sampling. Smaller alpha gives low-resource languages")
    print("      a larger share, but makes them repeat themselves more.")
    print("      Pick alpha from this table, together with the V4 rows of the cost table in notebooks/data_overview.ipynb §8.")


if __name__ == "__main__":
    main()
