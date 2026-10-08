#!/usr/bin/env python3
"""RQ1, "which language features make a helper effective?": measure each helper against each target
language and put the measurements next to the gain that helper gave in the RQ1 runs.

    python scripts/language_features_rq1.py --results results_main_h100 --out results_main_h100/language_features.md

Measures, for target T (be, az) and helper H (uk, ru_matched, ru, tr):

    script        writing system of T and H
    pairs         H-en training pairs the run used
    URIEL syn/pho/gen
                  typological distance (cosine distance of lang2vec vectors: syntax_knn, phonology_knn,
                  fam); 0 = identical, 1 = unrelated. Says nothing about the corpus, only the languages.
    chrF(T,H)     character n-gram overlap between a T sentence and its H translation, on positions where
                  T and every helper have text, so all helpers are measured on the same content
    word id       share of T word tokens that appear unchanged in the aligned H sentence (cognates, names)
    lex cov       share of T test word tokens that occur anywhere in H's training text: what the model
                  could have seen of T's words through this helper. Depends on corpus size too.
    NLLB pieces   share of T test subword tokens (NLLB's own SentencePiece, multi-letter pieces only, so
                  shared single letters and punctuation do not count) that also occur in H's training
                  text: the subword channel the fine-tuned model actually had
    gain          BLEU change against `none`, averaged over seeds, into and out of English

The NLLB vocabulary is the one the RQ1 models use, which is why it is measured here rather than the
WP0 vocabularies (V1-V5), which belong to the from-scratch track.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import TedData, align  # noqa: E402
from src.helper_training import build_mixture  # noqa: E402

SCRIPT = {"be": "Cyrillic", "uk": "Cyrillic", "ru": "Cyrillic", "az": "Latin", "tr": "Latin"}
URIEL = {"be": "bel", "uk": "ukr", "ru": "rus", "az": "azj", "tr": "tur"}
HELPERS = ["uk", "ru_matched", "ru", "tr"]
LANG = {"uk": "uk", "ru_matched": "ru", "ru": "ru", "tr": "tr"}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)  # letters only


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def uriel(a: str, b: str) -> dict[str, float]:
    import lang2vec.lang2vec as l2v

    out = {}
    for name, fs in (("syn", "syntax_knn"), ("pho", "phonology_knn"), ("gen", "fam")):
        f = l2v.get_features([URIEL[a], URIEL[b]], fs)
        x, y = (np.array([float(v) if v != "--" else 0.0 for v in f[URIEL[c]]]) for c in (a, b))
        out[name] = 1 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    return out


def gains(results: Path, target: str) -> dict[str, tuple[float, float]]:
    """{helper: (BLEU gain into English, BLEU gain out of English)} averaged over seeds, against none."""
    def mean(run: str, d: str) -> float:
        files = sorted((results / "metrics").glob(f"nllb_helper_{run}_s[0-9]*/scores.json"))
        files = [f for f in files if re.fullmatch(rf"nllb_helper_{run}_s\d+", f.parent.name)]
        return statistics.mean(json.loads(f.read_text())[d]["BLEU"] for f in files)
    into, out = f"{target}-en", f"en-{target}"
    return {h: (mean(h, into) - mean("none", into), mean(h, out) - mean("none", out)) for h in HELPERS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, help="also write the table to this markdown file")
    args = ap.parse_args()

    import sentencepiece as spm
    from huggingface_hub import hf_hub_download
    from sacrebleu.metrics import CHRF

    sp = spm.SentencePieceProcessor(model_file=hf_hub_download("facebook/nllb-200-distilled-600M",
                                                               "sentencepiece.bpe.model"))
    chrf = CHRF()
    ted = TedData(download=False)

    # helper training text exactly as the runs used it (ru_matched: the seed-42 subset)
    helper_text = {"uk": [h for _, h, _ in ted.pairs("train", "uk", "en")],
                   "ru": [h for _, h, _ in ted.pairs("train", "ru", "en")],
                   "tr": [h for _, h, _ in ted.pairs("train", "tr", "en")],
                   "ru_matched": [r["src"] for r in build_mixture(ted, "ru_matched", seed=42)["ru"]
                                  if r["direction"] == "ru-en"]}
    vocab = {h: {w for t in texts for w in words(t)} for h, texts in helper_text.items()}
    pieces = {h: {p for ids in sp.encode(texts, out_type=str) for p in ids} for h, texts in helper_text.items()}

    def content_pieces(text_list):
        return [p for ids in sp.encode(text_list, out_type=str) for p in ids
                if len(p.lstrip("▁")) > 1 and p.lstrip("▁").isalpha()]

    out = []
    for target in ("be", "az"):
        test = [t for _, t, _ in ted.pairs("test", target, "en")]
        test_words = [w for t in test for w in words(t)]
        test_pieces = content_pieces(test)
        # same content for every helper: positions where the target and all helper languages have text
        pos = align.positions("train", [target, "uk", "ru", "tr"])
        raw = {lang: align._raw("train", lang) for lang in (target, "uk", "ru", "tr")}
        par = {lang: [raw[lang][i].strip() for i in pos] for lang in raw}
        g = gains(args.results, target)

        out.append(f"\n### {target} ({SCRIPT[target]}); parallel measures on {len(pos):,} sentences "
                   f"where {target}, uk, ru and tr all have text\n")
        out.append("| helper | script | pairs | URIEL syn | URIEL pho | URIEL gen | chrF(T,H) | word id "
                   f"| lex cov | NLLB pieces | gain {target}-en | gain en-{target} |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for h in HELPERS:
            lang = LANG[h]
            u = uriel(target, lang)
            c = chrf.corpus_score(par[target], [par[lang]]).score
            same = sum(w in set(words(hs)) for ts, hs in zip(par[target], par[lang]) for w in words(ts))
            total = sum(len(words(ts)) for ts in par[target])
            cov = sum(w in vocab[h] for w in test_words) / len(test_words)
            pc = sum(p in pieces[h] for p in test_pieces) / len(test_pieces)
            out.append(f"| {h} | {SCRIPT[lang]} | {len(helper_text[h]):,} | {u['syn']:.2f} | {u['pho']:.2f} "
                       f"| {u['gen']:.2f} | {c:.1f} | {same / total:.1%} | {cov:.1%} | {pc:.1%} "
                       f"| {g[h][0]:+.2f} | {g[h][1]:+.2f} |")
    out.append("\nURIEL: cosine distance, 0 = identical. Parallel measures (chrF, word id) use the same sentences "
               "for every helper; ru and ru_matched are the same language, so only lex cov and NLLB pieces "
               "(which depend on the training text) can differ between them.")
    text = "\n".join(out)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
