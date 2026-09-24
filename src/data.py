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
to.** Training and evaluation code can take `TedData` instead: the same path, plus fetching and
unpacking the corpus when it is missing, NLLB-style examples, the test references every system has
to be scored against, and segmentation with the WP0 vocabularies. `align.py` is the project's single data entry point: it exposes *fetch by position* only, so
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


class TedData:
    """The authoritative path, packaged for training and evaluation code, on a laptop or on Colab.

    Creating one makes sure `data/raw/` exists: `TED.tar` is downloaded from the hub only if it is
    missing (to `data/TED.tar`, where `baselines_nllb.ipynb` already puts it) and unpacked through
    `align.py` only if a raw file is missing. Everything after that reads through `align.py`, so the
    pairs are the ones in `data/processed/parallel/` and `verify.py` covers them.

        ted = TedData()
        ted.pairs("train", "az", "en")                        # [(position, az, en), ...]
        ted.examples("test", ["az-en", "en-az"], codes)       # NLLB-style rows
        ted.references("en-az")                               # what every system's en-az.ref must hold
        ted.check_references("results/translations/nllb_basemodel")
        ted.segment(texts, "az", "V3_8000_unigram")           # WP0 subword pieces
    """

    HUB_REPO = "davidstap/ted_talks"

    def __init__(self, download: bool = True) -> None:
        self._pairs: dict[tuple[str, str, str], list[tuple[int, str, str]]] = {}
        self._tokenizers: dict = {}
        missing = sorted({lang for split in align.SPLIT_DIR for lang in align.PROJECT_LANGS
                          if not (align.RAW / split / f"{lang}.txt").exists()})
        if not missing:
            return
        if not align.TAR.exists():
            if not download:
                raise FileNotFoundError(f"{align.TAR} is missing: run `make data/TED.tar`, or pass download=True")
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=self.HUB_REPO, filename="data/TED.tar", repo_type="dataset", local_dir=align.ROOT)
        align.extract(missing)

    @staticmethod
    def _check(split: str, *langs: str) -> None:
        if split not in align.SPLIT_DIR:
            raise ValueError(f"unknown split {split!r}: use one of {list(align.SPLIT_DIR)}")
        for lang in langs:
            if lang not in align.PROJECT_LANGS:
                raise ValueError(f"unknown language {lang!r}: use one of {align.PROJECT_LANGS}")
        if len(set(langs)) != len(langs):
            raise ValueError(f"a pair needs two different languages, got {langs}")

    def pairs(self, split: str, src: str, tgt: str) -> list[tuple[int, str, str]]:
        """`[(position, src_text, tgt_text), ...]` where both languages have text, as `get_language_pairs`.

        Each language pair is read once per split and kept; the reverse direction reuses it.
        """
        self._check(split, src, tgt)
        a, b = sorted((src, tgt))
        key = (split, a, b)
        if key not in self._pairs:
            self._pairs[key] = list(align.pairs(split, a, b))
        return list(self._pairs[key]) if src == a else [(pos, t, s) for pos, s, t in self._pairs[key]]

    def examples(self, split: str, directions, codes: dict[str, str] | None = None) -> list[dict]:
        """Translation examples `{"src", "tgt", "src_lang", "tgt_lang", "direction"}` for `directions`
        such as `["az-en", "en-az"]`; `codes` renames the languages, e.g. NLLB's `{"az": "azj_Latn"}`.

        Rows come per language pair, in the order the pairs first appear in `directions`, and each
        position gives one row per requested direction of its pair. So `["az-en", "en-az", "be-en",
        "en-be"]` reproduces `baselines_nllb.ipynb`'s JSONL line for line.
        """
        codes = codes or {}
        by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for direction in directions:
            src, tgt = direction.split("-")
            by_pair.setdefault(tuple(sorted((src, tgt))), []).append((src, tgt))
        rows = []
        for (a, b), wanted in by_pair.items():
            for _, a_text, b_text in self.pairs(split, a, b):
                text = {a: a_text, b: b_text}
                for src, tgt in wanted:
                    rows.append({"src": text[src], "tgt": text[tgt], "src_lang": codes.get(src, src),
                                 "tgt_lang": codes.get(tgt, tgt), "direction": f"{src}-{tgt}"})
        return rows

    def references(self, direction: str, split: str = "test") -> list[str]:
        """The target side of `direction` (e.g. `"en-az"`), one sentence per line: what a system's
        `<direction>.ref` has to contain for its score to be comparable with everyone else's."""
        src, tgt = direction.split("-")
        return [t for _, _, t in self.pairs(split, src, tgt)]

    def check_references(self, folder, split: str = "test") -> dict[str, str]:
        """For each `<src>-<tgt>.ref` in `folder`: `OK`, `OK: first n of m` (a quick run on a prefix),
        `DIFF`, or `unknown direction`."""
        verdicts = {}
        for ref in sorted(Path(folder).glob("*.ref")):
            try:
                ours = self.references(ref.stem, split)
            except ValueError:
                verdicts[ref.name] = "unknown direction"
                continue
            got = ref.read_text(encoding="utf-8").splitlines()
            if got == ours:
                verdicts[ref.name] = "OK"
            elif got and got == ours[:len(got)]:
                verdicts[ref.name] = f"OK: first {len(got)} of {len(ours)}"
            else:
                verdicts[ref.name] = "DIFF"
        return verdicts

    def segment(self, texts, lang: str, vocab: str = "V3_8000_unigram") -> list[list[str]]:
        """Subword pieces of each text under one WP0 vocabulary, `artifacts/vocabularies/<vocab>/`.

        V5 was trained on character-mapped text (`translit` in `train_vocabs.py`), so for V5 the text
        is mapped the same way first; without that V5 meets letters it never saw in training.
        """
        from tokenizers import Tokenizer

        if vocab not in self._tokenizers:
            path = align.ROOT / "artifacts" / "vocabularies" / vocab / "tokenizer.json"
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing: run `make vocabs`, or Part A of notebooks/data_overview.ipynb")
            self._tokenizers[vocab] = Tokenizer.from_file(str(path))
        texts = list(texts)
        if vocab.startswith("V5"):
            from train_vocabs import translit
            texts = [translit(t, lang) for t in texts]
        return [e.tokens for e in self._tokenizers[vocab].encode_batch(texts)]


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
