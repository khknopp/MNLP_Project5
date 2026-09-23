"""Data access for the project.

⚠️ **There are two ways in and they are NOT interchangeable for anything you will report.**

| | `get_language_pairs(split, src, tgt)` | `get_language_data(dataset, l1, l2)` |
|---|---|---|
| source | `data/raw/` + `align.py` | HuggingFace hub, streamed |
| needs a local `TED.tar` (673 MB) | yes | **no** |
| returns position numbers | **yes** | no |
| joinable to `data/processed/parallel/` and the pivot triples | **yes** | no |
| hashed / checkable with `verify.py` | **yes** | no |
| use it for | **anything that ends up in the report** | poking at the data, smoke tests |

**Use `get_language_pairs` (or `get_language_data_aligned`) unless you have a specific reason not
to.** `align.py` is the project's single data entry point: it exposes *fetch by position* only, so
a caller cannot accidentally drop the blank lines that carry the alignment. The HuggingFace path
below returns plain pairs with no positions, which is convenient but unverifiable — two people can
run it and get results that look comparable and are not.

The two agree today: measured 2026-09-16, az-en train is 5,946 pairs on both paths, at identical
positions (`neulab/ted_multi`'s splits are 258,098 / 6,049 / 7,213, the same as `TED.tar`'s). That
is a **measured fact about the current data, not a guarantee** — HF can re-cut a dataset, and the
`revision` pin below is the only thing standing between us and that.

Split names differ between the two paths: this module's aligned path uses align.py's names
(`train` / `dev` / `test`), while the HF path returns `train` / `validation` / `test`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset

# align.py lives in src/preprocessing/. Put that directory on sys.path rather
# than using a relative import, so this works whether the caller was started from the repo
# root, from src/, from preprocessing/, or from a notebook.
_PREP = Path(__file__).resolve().parent / "preprocessing"
if str(_PREP) not in sys.path:
    sys.path.insert(0, str(_PREP))

import align  # noqa: E402  (must come after the sys.path line above)


# ------------------------------------------------------------------ authoritative path
def get_language_pairs(split: str, src: str, tgt: str) -> list[tuple[int, str, str]]:
    """Parallel pairs via `align.py`: `[(position, src_text, tgt_text), ...]`.

    `split` is one of `train` / `dev` / `test`. The position number is the raw line number in
    `data/raw/`, and it is the whole point of using this path: it lets a pair be joined to the
    pivot triples and to the frozen `data/processed/parallel/` artifacts, which carry a sha256 in
    `manifest.json` that `verify.py` can check.

    Raises `SystemExit` (from align.py) if `data/raw/` is missing — run `python align.py extract`
    first, then `python verify.py` to confirm the data matches the rest of the team's.
    """
    return list(align.pairs(split, src, tgt))


def get_language_data_aligned(split: str, src: str, tgt: str) -> list[tuple[str, str]]:
    """Same as `get_language_pairs`, without the positions: `[(src_text, tgt_text), ...]`.

    Drop-in shape for code that only wants the text (e.g. `Seq2SeqModel.fit`), so callers do not
    have to unpack a 3-tuple to use the authoritative path.
    """
    return [(s, t) for _, s, t in get_language_pairs(split, src, tgt)]


# --------------------------------------------------------- exploration path (unverifiable)
def get_dataset(dataset_name="neulab/ted_multi"):
    """Loads a HF dataset and returns train, val and test splits."""
    return load_dataset(dataset_name, revision="refs/convert/parquet", split=["train", "validation", "test"])


def get_language_data(dataset, lang_1, lang_2):
    """Returns a dataset filtered by language.

    ⚠️ **Exploration only — no position numbers, nothing hashed.** Kept because it needs no local
    `TED.tar` and is handy for a first look; do not build reported numbers on it. See the module
    docstring for the comparison with `get_language_pairs`.
    """
    pairs = []
    for line in dataset:
        langs = dict(zip(line["translations"]["language"], line["translations"]["translation"]))
        if lang_1 in langs and lang_2 in langs:
            pairs.append((langs[lang_1], langs[lang_2]))
    return pairs
