#!/usr/bin/env python3
"""
WP0 · Cost analysis: C1–C6 + a set of human-readable segmentation examples.

Answers three questions, none of which can be skipped:
    **How big is the cost?**  C1 fertility / C2 sequence inflation / C4 fragmentation rate
    **Where does the cost come from?** C5 vocabulary allocation share (how many slots az actually holds in the joint vocabulary)
    **What did it buy us?** C3 shared-piece share — **must be computed per helper language**
    **But is what got shared any good?** C6 shared-piece productivity — see below

⚠️ **C6 exists because C3 on its own cannot answer the question it looks like it answers.**
C3 says "N% of az's tokens land on pieces tr also uses". It does **not** say whether those
pieces are the morphology that carries meaning. A joint vocabulary that shares nothing but
punctuation, digits and stray Latin letters would score a perfectly respectable C3 while
transferring nothing — and nothing in C1–C5 would notice. data_overview.ipynb §7 states the criterion:

    "The criterion for 'recoverable' is whether the shared pieces land exactly on az's
     high-frequency affixes. If C3 is high but C1 stays bad, then what is shared is
     irrelevant pieces and sharing is pure dilution — which is itself the more
     interesting finding."

C6 operationalizes exactly that sentence: of az's token mass that **is** shared with a helper,
how much sits on **productive suffixes** (real morphology), on **junk** (pieces with no alphabetic
content — punctuation, digits, symbols), on **fragments** (single letters: script overlap, weak
but real), or on none of the three. Junk and fragments are kept apart on purpose — merging them
flips the conclusion, see the note above `classify_piece`.

Usage:
    python cost_analysis.py                    # all variants, all metrics
    python cost_analysis.py --variants V3_8000_bpe V3_8000_unigram
    python cost_analysis.py --split dev        # switch the eval set (default test)
    python cost_analysis.py --min-suffix-support 50   # stricter definition of "productive affix"

Outputs:
    out/cost_metrics.csv     the full C1–C6 table
    out/cost_report.md       markdown table + worked examples (paste straight into the report)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

HELPER = {"az": ["tr"], "be": ["uk", "ru"]}   # the helper languages for each low-resource language
TARGETS = ["az", "be"]
ALL6 = ["en", "az", "be", "tr", "uk", "ru"]


# --------------------------------------------------------- conventions
def applies_to(variant: str) -> list[str]:
    """Which languages a given vocabulary variant should be evaluated on.

    ⚠️ Monolingual vocabularies (V1_az / V1_be) **may only be evaluated on their
    own language** — using the az monolingual vocabulary to encode Belarusian is
    meaningless (every result is <unk>). This is an evaluation convention, not an
    optimization.
    """
    if variant == "V1_az":
        return ["az"]
    if variant == "V1_be":
        return ["be"]
    return list(TARGETS)


def baseline_tag(index: dict, lang: str, model: str | None = None) -> str | None:
    """The cost zero-point for a language = its own monolingual vocabulary.

    ⚠️ The model must be matched too: Unigram and BPE each have their own V1_az,
    so looking one up by language alone compares the two segmentation methods
    against each other and Δ stops being single-variable.
    """
    for tag, meta in index.items():
        if meta.get("variant") == f"V1_{lang}" and (model is None or meta.get("model") == model):
            return tag
    return None


# ------------------------------------------------------------- helpers
def load_tokenizer(d: Path) -> Tokenizer:
    return Tokenizer.from_file(str(d / "tokenizer.json"))


def marker_for(tok: Tokenizer) -> str:
    """Metaspace marks word starts with ▁, ByteLevel with Ġ."""
    s = json.dumps(tok.to_str())
    return "▁" if "Metaspace" in s else "Ġ"


def per_word_pieces(tok: Tokenizer, line: str, marker: str) -> list[int]:
    """Encode a sentence, then split it back by word-start markers into "how many pieces each word used"."""
    toks = tok.encode(line).tokens
    counts, cur = [], 0
    for i, t in enumerate(toks):
        if i > 0 and t.startswith(marker):
            counts.append(cur)
            cur = 0
        cur += 1
    if cur:
        counts.append(cur)
    return counts


def read_lines(p: Path) -> list[str]:
    """Non-empty, stripped lines — the same convention as train_vocabs.py / directionality.py."""
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def words(line: str) -> list[str]:
    return line.split()


def fmt(v, nd: int = 3) -> str:
    """Format an optional metric for the markdown report."""
    return "—" if v is None else f"{v:.{nd}f}"


def pieces_used(tok: Tokenizer, lines: list[str]) -> set[str]:
    out: set[str] = set()
    for ln in lines:
        out.update(tok.encode(ln).tokens)
    return out


def allocation_share(tok: Tokenizer, samples: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """C5: which language the slots in the joint vocabulary actually went to.

    For every piece in the vocabulary, count how often it occurs in the samples of
    the 6 languages:
      · alloc = the piece's **most frequent language** is this language → the slot counts as this language's
      · own   = the piece is **used only by this language**            → a slot this language holds exclusively
    Both denominators are the full vocabulary size, so a monolingual vocabulary
    should land near 1.0 (a handful of pure-symbol pieces have no frequency).

    This is the direct evidence for "where does the cost come from": joint training
    did not make az's representations worse, it **gave az fewer slots** — and only
    C5 can prove that claim.
    """
    freq = {l: Counter() for l in samples}
    for l, lines in samples.items():
        for ln in lines:
            freq[l].update(tok.encode(ln).tokens)

    vocab = tok.get_vocab()
    alloc = Counter()
    own = Counter()
    assigned = 0
    for p in vocab:
        counts = {l: freq[l].get(p, 0) for l in samples}
        tot = sum(counts.values())
        if tot == 0:
            continue                      # pure special symbols / never observed, not counted
        assigned += 1
        top = max(counts, key=lambda l: counts[l])
        alloc[top] += 1
        if sum(1 for c in counts.values() if c > 0) == 1:
            own[top] += 1

    n = len(vocab)
    return {
        l: {
            "C5_alloc_share": round(alloc.get(l, 0) / n, 4),
            "C5_own_share": round(own.get(l, 0) / n, 4),
            "C5_assigned_pieces": assigned,
            "C5_vocab_size": n,
        }
        for l in samples
    }


# ------------------------------------------------- C6: but is the shared material any good?
# ⚠️ Measured pitfall (2026-09-16): a first version lumped single alphabetic characters in with
# punctuation under one "junk" bucket. That inverted the finding — for V3/unigram, az's shared
# token mass is 13.6% punctuation/digits but 29.3% single letters, so "junk 0.43 > useful 0.27"
# was an artifact of the bucketing, and under the honest split real morphology (0.27) is twice
# the true noise (0.136). A single letter is script overlap, not dilution. Keep them apart.
MIN_AFFIX_CHARS = 2     # shorter than this, a piece cannot be morphology — it is a fragment


def suffix_support(tok: Tokenizer, lines: list[str], marker: str, max_len: int = 20) -> Counter:
    """How many **distinct word types** each piece serves as an ending for.

    This is a type-based notion of productivity, and it is what separates real morphology from
    coincidence: az/tr morphology is **suffixal**, so `-lər` is a plural marker precisely
    because hundreds of different word types end in it.

    ⚠️ It is deliberately not the whole answer, which is why `classify_piece` exists: pieces
    like `.` or `20` also "end" a great many word types. Productivity alone would crown them
    the most productive affixes in the language.

    Counted over the target language's **training** text, not the eval split — 900 sentences
    is far too little to estimate a type count stably. This is a statistic about the
    vocabulary, not an evaluation number, so using train text contaminates nothing (the same
    reasoning as C3's helper side).

    Only **bare** (marker-less) strings are counted, because a piece carrying the word-start
    marker can never be a suffix — words are whitespace-delimited, so `▁` never occurs inside one.
    """
    freq: Counter = Counter()
    for ln in lines:
        freq.update(ln.split())          # type frequencies; we count types, not tokens

    in_vocab = set(tok.get_vocab())
    support: Counter = Counter()
    for w in freq:
        # every **proper** character-suffix of the word (n < len(w): a word ending in itself
        # is a whole word, not an affix)
        for n in range(2, min(len(w) - 1, max_len) + 1):
            s = w[-n:]
            if s in in_vocab:
                support[s] += 1
    return support


def classify_piece(p: str, marker: str, support: Counter, min_support: int) -> str:
    """Bucket a piece as `affix` / `fragment` / `junk` / `other`. All of C6 rests on these four.

    ⚠️ This is a **linguistic judgment**, not an implementation detail — the thresholds should be
    confirmed with the group before submitting (same status as `TRANSLIT` in train_vocabs.py).

      · `junk`     — no alphabetic character at all (`▁,` · `.` · `20` · `(…)`). Two languages in
                     one script share these for free; a high junk share is dilution, not transfer.
      · `fragment` — exactly one alphabetic character. **Deliberately *not* junk:** a shared
                     single letter is script-level overlap, which is real (if weak) transfer, and
                     folding it into junk is what made an earlier version of this metric claim
                     the opposite of what the data says.
      · `affix`    — ends >= `min_support` distinct word types. **This is the transfer channel
                     data_overview.ipynb §7 asks about.**
      · `other`    — stems, whole words, and every word-initial piece (which can never be a
                     suffix, so it lands here by construction — looked up as-is, never
                     marker-stripped, so `▁göz` is never credited with `göz`'s suffix count).
    """
    bare = p[len(marker):] if p.startswith(marker) else p
    if not any(ch.isalpha() for ch in bare):
        return "junk"
    if len(bare) < MIN_AFFIX_CHARS:
        return "fragment"
    if support.get(p, 0) >= min_support:
        return "affix"
    return "other"


# ------------------------------------------------------------ main flow
def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=root / "data" / "processed")
    ap.add_argument("--vocab-dir", type=Path, default=root / "out" / "vocabs")
    ap.add_argument("--out-dir", type=Path, default=root / "out")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--max-lines", type=int, default=0, help="0=all lines; cap it when debugging")
    ap.add_argument("--examples", type=int, default=12, help="how many segmentation examples to export")
    ap.add_argument("--helper-lines", type=int, default=50000,
                    help="how many leading sentences of the helper-language training text define the shared-piece set")
    ap.add_argument("--alloc-frac", type=float, default=1.0,
                    help="⚠️ fraction of each language's training corpus that C5 samples. Default "
                         "1.0 = the whole corpus, which is the only reading that is stable. ANY "
                         "value below 1 changes the answer, sometimes by a factor of five — see "
                         "the measured table above the sampling code. Use it to iterate quickly, "
                         "never to produce a number for the report")
    ap.add_argument("--affix-lines", type=int, default=50000,
                    help="how many leading sentences of the target language's own training text "
                         "define how many word types each piece ends (C6)")
    ap.add_argument("--min-suffix-support", type=int, default=20,
                    help="a piece must end at least this many distinct word types to count as a "
                         "productive affix (C6). Stricter values shrink the 'affix' bucket")
    args = ap.parse_args()

    index = json.loads((args.vocab_dir / "index.json").read_text(encoding="utf-8"))
    tags = args.variants or sorted(index)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Eval text (the tokenizer was only trained on train → dev/test is clean)
    evals: dict[str, list[str]] = {}
    for lang in ["az", "be", "tr", "uk", "ru"]:
        p = args.data_dir / f"{lang}.{args.split}.txt"
        evals[lang] = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if args.max_lines:
            evals[lang] = evals[lang][: args.max_lines]

    # ⚠️ The denominator of C3 is "which pieces the helper language has used" — it is
    # determined by sampling its **training text**, not its small test set (the tr test
    # set is only 5k sentences and would systematically underestimate the shared share).
    helpers_text: dict[str, list[str]] = {}
    for lng in ["tr", "uk", "ru"]:
        p = args.data_dir / f"{lng}.train.txt"
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        helpers_text[lng] = lines[: args.helper_lines]

    # Samples used by C5: a fixed **fraction** of each language's training corpus.
    #
    # ⚠️ Measured pitfall (2026-09-16), and it produced a wrong headline number. C5 assigns every
    # slot to the language that uses it *most often in this sample*, so the sample's composition
    # decides the answer. An earlier version capped each language at a fixed **line count**, which
    # is not a harmless speed knob: the six corpora differ by 57x (az 5,946 vs en 258,098), so a
    # 20,000-line cap covered az entirely but en only 7.7%. That biases the result in a known
    # direction — it understates how far the large languages crowd the small ones out. Measured
    # for V3/az:  equal line count (4,509/lang) **8.0%** · (20,000/lang) **1.5%** · full **0.43%**.
    #
    # A fixed *fraction* is better than a fixed count, but it is **not** a safe shortcut either,
    # and the reason is subtle enough to be worth writing down: the sample is a **prefix**, not a
    # random draw, so the first 10% of the Azerbaijani corpus and the first 10% of the English one
    # cover different talks. The proportions are therefore not preserved. Measured:
    #     V1_az  0.127 (5%)  0.224 (10%)  0.394 (25%)  0.525 (50%)  0.610 (100%)   ← drifts
    #     V3_az  0.0043      0.0043      0.0041      0.0044      0.0043            ← stable
    # So V3/V5 tolerate subsampling but the monolingual zero-point does not, and a default that
    # is safe for one variant is wrong for another. The only defensible reading is the whole
    # corpus, which is also the faithful one: the vocabulary was trained on the full corpora, and
    # the mechanism being measured is "raw frequency fills a fixed-size vocabulary".
    # **Any C5 number must be quoted with its protocol** — the equal-count protocols give 1.5%
    # where the full corpora give 0.43%.
    alloc_samples: dict[str, list[str]] = {}
    for lng in ALL6:
        lines = read_lines(args.data_dir / f"{lng}.train.txt")
        alloc_samples[lng] = lines[: max(1, int(len(lines) * args.alloc_frac))]
    print("C5 sampling: " + "  ".join(f"{k}={len(v):,}" for k, v in alloc_samples.items())
          + f"   ({args.alloc_frac:.0%} of each corpus)")

    # C6: the text whose word types decide which pieces count as productive suffixes.
    # Each target language uses its own training text — see suffix_support() for why this
    # is a vocabulary statistic rather than a contaminated evaluation number.
    affix_text: dict[str, list[str]] = {}
    for lng in TARGETS:
        p = args.data_dir / f"{lng}.train.txt"
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        affix_text[lng] = lines[: args.affix_lines]
    print("C6 affix reference: " + "  ".join(f"{k}={len(v):,}" for k, v in affix_text.items()))

    rows: list[dict] = []
    for tag in tags:
        if tag not in index:
            print(f"  skipping {tag} (not in index.json)")
            continue
        tok = load_tokenizer(args.vocab_dir / tag)
        marker = marker_for(tok)
        langs_here = applies_to(index[tag]["variant"])
        mono = index[tag]["variant"].startswith("V1")

        # The set of pieces the helper languages "have used". Under a monolingual
        # vocabulary the helper languages are all <unk>, which is meaningless → skip it.
        helper_pieces: dict[str, set[str]] = (
            {} if mono else {lng: pieces_used(tok, helpers_text[lng]) for lng in helpers_text}
        )
        c5 = allocation_share(tok, alloc_samples)

        for lang in langs_here:
            tot_words = tot_pieces = 0
            frag = 0
            tot_tokens = 0
            shared = Counter()               # how many tokens each helper language hits on its own
            shared_any = 0                   # hit by at least one helper (union reading, reference only)
            union = set()
            for h in HELPER[lang]:
                union |= helper_pieces.get(h, set())
            # C6 is per-vocabulary (every variant has its own pieces), so it is recomputed here.
            # Under a monolingual vocabulary nothing is shared, so every token lands in the
            # "not shared" bucket — which is exactly the reference reading C6 needs.
            support = suffix_support(tok, affix_text[lang], marker)
            cls_shared: Counter = Counter()      # tokens shared with a helper, by piece class
            cls_nonshared: Counter = Counter()   # tokens not shared with any helper, by piece class
            own_pieces: set[str] = set()
            for ln in evals[lang]:
                ws = words(ln)
                c = per_word_pieces(tok, ln, marker)
                if len(c) != len(ws):        # fallback for when the marker method breaks, so counts never go silently wrong
                    c = c + [0] * (len(ws) - len(c))
                tot_words += len(ws)
                tot_pieces += sum(c)
                frag += sum(1 for x in c if x >= 4)
                toks = tok.encode(ln).tokens
                tot_tokens += len(toks)
                own_pieces.update(toks)
                for h in HELPER[lang]:
                    hp = helper_pieces.get(h, set())
                    shared[h] += sum(1 for t in toks if t in hp)
                shared_any += sum(1 for t in toks if t in union)
                for t in toks:
                    cls = classify_piece(t, marker, support, args.min_suffix_support)
                    (cls_shared if t in union else cls_nonshared)[cls] += 1

            row = {
                "tag": tag,
                "variant": index[tag]["variant"],
                "alpha": index[tag].get("alpha"),
                "lang": lang,
                "model": index[tag]["model"],
                "vocab_size": index[tag]["vocab_size_actual"],
                "C1_fertility": round(tot_pieces / tot_words, 3) if tot_words else 0,
                "C2_tokens_total": tot_tokens,
                # Union reading: reference only. Any cross-language comparison must use C3_share_<helper>
                "C3_any_share": (
                    None if (mono or not tot_tokens) else round(shared_any / tot_tokens, 4)
                ),
                "C4_frag4_rate": round(frag / tot_words, 4) if tot_words else 0,
                "own_pieces": len(own_pieces),
            }
            # C6. The headline is `useful_share_of_shared`: of the token mass that is shared
            # with a helper at all, how much sits on real morphology. The junk share is its
            # counterpart — high junk is data_overview.ipynb §7's "sharing is pure dilution" branch.
            # `useful_share_of_nonshared` is the control: it says how much morphology az
            # keeps for itself, so a shared share *below* it would mean sharing is actively
            # diverting tokens away from productive pieces.
            sh_tot = sum(cls_shared.values())
            ns_tot = sum(cls_nonshared.values())
            row["C6_shared_tokens"] = sh_tot
            row["C6_useful_share_of_shared"] = round(cls_shared["affix"] / sh_tot, 4) if sh_tot else None
            row["C6_fragment_share_of_shared"] = round(cls_shared["fragment"] / sh_tot, 4) if sh_tot else None
            row["C6_junk_share_of_shared"] = round(cls_shared["junk"] / sh_tot, 4) if sh_tot else None
            row["C6_useful_share_of_nonshared"] = round(cls_nonshared["affix"] / ns_tot, 4) if ns_tot else None
            # C3 is reported per helper language — the union cannot be compared across
            # languages (az has 1 helper, be has 2)
            for h in ["tr", "uk", "ru"]:
                row[f"C3_share_{h}"] = (
                    None if (mono or not tot_tokens or h not in HELPER[lang])
                    else round(shared[h] / tot_tokens, 4)
                )
            row.update(c5[lang])
            rows.append(row)
            c3s = " ".join(
                f"{h}={row[f'C3_share_{h}']:.3f}" for h in HELPER[lang]
                if row[f"C3_share_{h}"] is not None
            )
            c6 = ("      —      " if row["C6_useful_share_of_shared"] is None
                  else f"{row['C6_useful_share_of_shared']:.3f}/"
                       f"{row['C6_fragment_share_of_shared']:.3f}/"
                       f"{row['C6_junk_share_of_shared']:.3f}")
            print(f"  {tag:20s} {lang}  C1={row['C1_fertility']:.3f}  {c3s}  "
                  f"C4={row['C4_frag4_rate']:.3f}  C5alloc={row['C5_alloc_share']:.3f}  "
                  f"C6(affix/frag/junk)={c6}")

    # ---- C2 relative inflation: on the same sentences, each variant's token count ÷
    # that language's own monolingual zero-point for the same segmentation model
    base = {
        (r["lang"], r["model"]): r
        for r in rows
        if r["tag"] == baseline_tag(index, r["lang"], r["model"])
    }
    for r in rows:
        b = base.get((r["lang"], r["model"]))
        r["C2_inflation_vs_baseline"] = (
            round(r["C2_tokens_total"] / b["C2_tokens_total"], 3) if b and b["C2_tokens_total"] else None
        )

    with open(args.out_dir / "cost_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- Worked examples (the error analysis the advisor explicitly asked for),
    # broken out per segmentation model
    ex_md: list[str] = []
    for mdl in sorted({index[t]["model"] for t in tags if t in index}):
        v3c = [t for t in tags if t in index and index[t]["variant"] == "V3"
               and index[t]["model"] == mdl]
        if not v3c:
            continue
        v3tag = v3c[0]
        t3 = load_tokenizer(args.vocab_dir / v3tag)
        m3 = marker_for(t3)
        for lang in TARGETS:
            btag = baseline_tag(index, lang, mdl)
            if btag is None or btag not in tags:
                continue
            tb = load_tokenizer(args.vocab_dir / btag)
            mb = marker_for(tb)
            seen, cost = set(), []
            for ln in evals[lang]:
                c1 = per_word_pieces(tb, ln, mb)
                c3 = per_word_pieces(t3, ln, m3)
                if len(c1) != len(c3):
                    continue
                for wd, a, b in zip(words(ln), c1, c3):
                    if a <= 1 and b >= 3 and wd not in seen:
                        seen.add(wd)
                        cost.append((b, f"| `{wd}` | {a} | {b} |"))
                if len(cost) >= args.examples * 2:
                    break
            cost.sort(key=lambda x: -x[0])          # the most fragmented ones first
            ex_md.append(f"\n### {lang} ({mdl}): 1 piece monolingually → shattered under the joint vocabulary (the cost made concrete)\n")
            ex_md.append(f"| word | {btag} | {v3tag} |")
            ex_md.append("|---|---|---|")
            ex_md.extend([c[1] for c in cost[: args.examples]] or ["| (no such examples found) | | |"])

    md = [
        f"# WP0 cost analysis ({args.split} split)",
        "",
        "**Zero-point**: each language's own monolingual vocabulary `V1_az` / `V1_be`, and it "
        "**must be the same segmentation model** (Unigram inflation can only be divided by a "
        "Unigram zero-point).",
        "",
        "| Metric | Meaning | How to read it |",
        "|---|---|---|",
        "| **C1** fertility | piece / whitespace word | higher = more fragmented representation |",
        "| **C2** inflation | this variant's token count ÷ the monolingual zero-point | ×1.76 = the same content costs 76% more sequence length |",
        "| **C3** shared share | fraction of tokens landing on pieces that **some helper language** also used | **read it per helper**, do not compare across languages |",
        "| **C4** fragmentation rate | fraction of words cut into ≥4 pieces | tail risk, worst in agglutinative languages |",
        "| **C5** allocation share | fraction of vocabulary slots whose most frequent language is this language | **the mechanism of the cost** — but read the caveat below: this is *absolute frequency*, not ownership |",
        "| **C6** shared-piece productivity | of the tokens shared with a helper: `useful` = productive suffixes, `frag` = single letters (script overlap), `junk` = no alphabetic content | **whether the sharing is worth anything** — high C3 with high C6 junk = dilution, not transfer |",
        "",
        "| Vocabulary | Segmentation | Language | Size | **C1** | **C2** | **C3** | **C4** | **C5 alloc** | **C5 own** | **C6 useful** | **C6 frag** | **C6 junk** |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c3 = " / ".join(
            f"{h}:{r[f'C3_share_{h}']:.3f}" for h in HELPER[r["lang"]]
            if r.get(f"C3_share_{h}") is not None
        ) or "—"
        f2 = lambda k: "—" if r.get(k) is None else f"{r[k]:.3f}"
        md.append(
            f"| {r['tag']} | {r['model']} | {r['lang']} | {r['vocab_size']:,} | "
            f"{r['C1_fertility']:.3f} | "
            f"{'—' if r['C2_inflation_vs_baseline'] is None else r['C2_inflation_vs_baseline']} | "
            f"{c3} | {r['C4_frag4_rate']:.3f} | {f2('C5_alloc_share')} | {f2('C5_own_share')} | "
            f"{f2('C6_useful_share_of_shared')} | {f2('C6_fragment_share_of_shared')} | "
            f"{f2('C6_junk_share_of_shared')} |"
        )
    md += ["", "> **C1 and C5 must be read together**: a rising C1 is only the symptom, C5 is what explains the mechanism —",
           "> joint training did not make az's representation quality worse, it **gave az fewer slots in a fixed-size vocabulary**.",
           "> **C3 must be read per helper**: az has only tr as a helper, be has uk+ru (two), so under",
           "> the union reading be's C3 is higher by construction — comparing it across languages is fake.", "",
           "> ⚠️ **C5 measures *absolute frequency* dominance, not ownership.** A slot is credited to",
           "> whichever language uses it most often, so a large corpus wins slots it has no real claim",
           "> to. The proof is the zero-point: under `V1_az` — trained on Azerbaijani text alone — az",
           "> reaches only 61%, not the ~1.0 the metric was designed to give. Read it as \"whose",
           "> frequency dominates this slot\", never as \"how many slots this language owns\".",
           ">",
           "> ⚠️ **And C5 is the one metric whose value depends on how the corpus is sampled** — the",
           "> same quantity reads 8.0% / 1.5% / 0.43% for equal-line-count, 20k-line and whole-corpus",
           "> sampling (V3/az). Only the whole-corpus reading is stable, and it is the default.",
           "> **Never quote a C5 number without its protocol.** See `out/sensitivity.md`.", ""]

    # ---- C6: the reading data_overview.ipynb §7 asks for. C3 alone cannot tell "the shared pieces carry
    # morphology" apart from "the shared pieces are punctuation" — both score well on C3.
    md += ["---", "", "## C6 · Is the shared material worth anything?", "",
           "data_overview.ipynb §7 states the criterion:", "",
           "> \"The criterion for 'recoverable' is whether the shared pieces land exactly on az's",
           "> high-frequency affixes. If C3 is high but C1 stays bad, then what is shared is",
           "> irrelevant pieces and sharing is pure dilution — which is itself the more interesting",
           "> finding.\"", "",
           f"A piece counts as a **productive affix** if it ends at least **{args.min_suffix_support}** "
           f"distinct word types in that language's own training text (first {args.affix_lines:,} "
           f"sentences). The remaining buckets are **fragments** (a single alphabetic character — "
           f"script overlap: weak transfer, but transfer) and **junk** (no alphabetic character at "
           f"all — punctuation, digits, symbols: none). All are shares of the **shared** token mass; "
           f"`useful (not shared)` is the control — how much morphology the language keeps for "
           f"itself — and for the monolingual V1 there is no sharing at all, so that column *is* "
           f"its full reading.", "",
           "**How to read it.** High `C3` with high `C6 useful` means the transfer channel carries "
           "real morphology. High `C3` with high `C6 junk` means the joint vocabulary is sharing "
           "punctuation and digits — dilution dressed up as transfer, and the more interesting "
           "result of the two. **`frag` must not be folded into `junk`**: doing so inverts the "
           "conclusion, because single letters are the largest single component of the shared "
           "mass and they are script overlap, not noise.", "",
           "⚠️ The thresholds are a linguistic judgment, not an implementation detail — confirm them "
           "with the group before they go into the report (same status as `TRANSLIT` in "
           "`train_vocabs.py`)."]
    for mdl in sorted({index[t]["model"] for t in tags if t in index}):
        for lng in TARGETS:
            sel = [r for r in rows if r["model"] == mdl and r["lang"] == lng]
            if not any(r["C6_shared_tokens"] for r in sel):
                continue
            md += ["", f"### {mdl} · {lng}", "",
                   "| Vocabulary | C3 | shared tokens | **C6 useful** | **C6 frag** | **C6 junk** | useful (not shared) |",
                   "|---|---|---|---|---|---|---|"]
            for r in sel:
                helpers = " / ".join(
                    f"{h}:{r[f'C3_share_{h}']:.3f}" for h in HELPER[lng]
                    if r.get(f"C3_share_{h}") is not None
                ) or "—"
                md.append(
                    f"| {r['tag']} | {helpers} | {r['C6_shared_tokens']:,} | "
                    f"{fmt(r['C6_useful_share_of_shared'])} | {fmt(r['C6_fragment_share_of_shared'])} | "
                    f"{fmt(r['C6_junk_share_of_shared'])} | "
                    f"{fmt(r['C6_useful_share_of_nonshared'])} |"
                )

    md += ex_md
    (args.out_dir / "cost_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n→ {args.out_dir/'cost_metrics.csv'}\n→ {args.out_dir/'cost_report.md'}")


if __name__ == "__main__":
    main()
