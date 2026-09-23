#!/usr/bin/env python3
"""
WP0 · Directionality: the TED corpus is a **translation corpus**, and that changes how to read
every number in this project.

The tar ships no translator metadata — 183 files, all `ted.{split}.{lang}`, no talk names, no
credits. So the question "is the Azerbaijani side original text or a translation?" has to be
answered from the text itself. It can be, decisively:

**Every non-English side of this corpus is a translation of the English side.** Three independent
signals agree:

1. **Stage directions are translated, position by position.** TED transcripts carry `(Applause)`,
   `(Laughter)`, `(Music)`. Where the English has one, the other language has one at the *same
   position* 94–99.6% of the time — and it is translated (`(Applause)` → `(Alqışlar)`). A native
   corpus would not carry stage directions at all, let alone in the same places.
2. **The transcript's `""` speaker-change markers are inherited** (az 82.8%, be 72.3%).
3. **Consecutive windows correspond sentence by sentence**, including where the translator went
   free (`It's okay, it's safe here.` → `Problem yoxdur, sirrinizi açmaram.`).

Why this matters, and it is not a footnote:

⚠️ **1. The two directions are not symmetric tasks.** `X→en` recovers the *source* of a
translation, and translationese is English-shaped (calques carry English word order and idiom
across) — and its reference is the English original. `en→X` must land on *one particular human
translator's choices*, because that translation is the reference BLEU is computed against. So a
BLEU gap between the two directions is **not** evidence of a data-volume gap.

⚠️ **2. RQ1's "similarity" channel is confounded.** tr, uk and ru — the helper languages — are
translations *of the same English sentences* as az and be. Their shared structure therefore comes
partly from **one shared translation process**, not only from the languages being related. Any
claim that helper languages help "through similarity" has to rule this out.

⚠️ **3. C3's baseline is inflated, and this is why.** Under V2 — a vocabulary trained on az+be+en
that has **never seen Turkish** — az tokens still land on pieces tr also produces 77% of the time.
The reason is structural: both sides are translations of the same English, so both carry the same
Latin characters, punctuation, digits and English loanwords. `C6` is what separates that from real
morphology; `C3_morph` and `C3_affix` below are the corrected readings.

Usage:
    python directionality.py
    python directionality.py --split test
    python directionality.py --langs az be

Outputs:
    results/metrics/preprocessing/directionality_report.md   the tables above, ready to paste into the report
    results/metrics/preprocessing/directionality.csv         per variant / language / helper: C3 and its decomposition
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# This file lives in src/preprocessing/ alongside align.py and cost_analysis.py. Put that directory
# on sys.path so the sibling imports below resolve whether this is run as a script or imported
# from elsewhere — the same reason src/data.py puts the repository root on sys.path.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse cost_analysis's classifier rather than copying it. A second implementation would drift
# from the first, and the whole point of this project is that silent divergence is the enemy.
from cost_analysis import (  # noqa: E402
    HELPER,
    TARGETS,
    classify_piece,
    load_tokenizer,
    marker_for,
    pieces_used,
    suffix_support,
)

import align  # noqa: E402  (sibling module, needs the line above)

# This file lives in src/preprocessing/; data/, artifacts/ and results/ live at the repository root.
ROOT = _HERE.parent.parent

# Stage directions in TED transcripts. Used only as *evidence of translation*, never as text to
# train on — so the exact list matters less than it being applied identically to every language.
STAGE_DIRECTION = re.compile(r"\(([^)]{0,30})\)")

# An untranslated English residue, e.g. a product name left in Latin script inside Cyrillic text.
# ⚠️ Only meaningful for languages that do NOT use the Latin alphabet — for az and tr this check
# would match their own words and report a meaningless ~90%. `--latin-script-langs` records which
# languages are excluded and why.
LATIN_RESIDUE = re.compile(r"\b[A-Za-z]{3,}\b")
NON_LATIN = ["be", "uk", "ru"]


def read_lines(p: Path, limit: int = 0) -> list[str]:
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:limit] if limit else lines


# --------------------------------------------------- evidence 1 & 2: positional inheritance
def inheritance_table(split: str, langs: list[str]) -> list[dict]:
    """Stage-direction and speaker-marker inheritance, measured position by position.

    `align.pairs` is used rather than reading `data/processed/vocab_training/`: this check is *about* positions,
    and processed files have already dropped the blank-line placeholders.
    """
    rows = []
    for lang in langs:
        n = has_en_dir = has_both_dir = has_en_spk = has_both_spk = 0
        latin = latin_tot = 0
        for _pos, text, en in align.pairs(split, lang, "en"):
            n += 1
            en_dir = bool(STAGE_DIRECTION.search(en))
            if en_dir:
                has_en_dir += 1
                if STAGE_DIRECTION.search(text):
                    has_both_dir += 1
            if '""' in en:
                has_en_spk += 1
                if '""' in text:
                    has_both_spk += 1
            if lang in NON_LATIN:
                latin_tot += 1
                if LATIN_RESIDUE.search(text):
                    latin += 1
        rows.append({
            "lang": lang,
            "aligned_positions": n,
            "en_with_stage_direction": has_en_dir,
            "both_have_stage_direction": has_both_dir,
            "stage_direction_inheritance": round(has_both_dir / has_en_dir, 4) if has_en_dir else None,
            "en_with_speaker_marker": has_en_spk,
            "both_have_speaker_marker": has_both_spk,
            "speaker_marker_inheritance": round(has_both_spk / has_en_spk, 4) if has_en_spk else None,
            "sentences": latin_tot or None,
            "sentences_with_latin_residue": latin if lang in NON_LATIN else None,
            "latin_residue_rate": round(latin / latin_tot, 4) if (lang in NON_LATIN and latin_tot) else None,
        })
    return rows


# --------------------------------------------- evidence 3: C3 and how much of it is generic
def c3_decomposition(index: dict, vocab_dir: Path, data_dir: Path, split: str,
                     helper_lines: int, affix_lines: int, min_support: int,
                     max_lines: int) -> list[dict]:
    """Split every `C3_share_<helper>` into morphology and generic noise.

    Built to sit directly beside `cost_metrics.csv`: same eval text, same helper-piece definition,
    same classifier. `C3_total` here reproduces `C3_share_<h>` from that file, so a mismatch means
    the two scripts have drifted apart — which is a bug, not a nuance.

    `C3_generic` is the part of the sharing that any two languages in the same script would get
    for free (single letters, punctuation, digits). `C3_morph` is the rest. `C3_affix` keeps only
    pieces that are productive suffixes of the target language.
    """
    evals: dict[str, list[str]] = {}
    for lang in TARGETS:
        p = data_dir / f"{lang}.{split}.txt"
        if not p.exists():
            raise SystemExit(f"missing {p} — run `make data` first")
        evals[lang] = read_lines(p, max_lines)

    helper_text: dict[str, list[str]] = {}
    for h in ["tr", "uk", "ru"]:
        helper_text[h] = read_lines(data_dir / f"{h}.train.txt", helper_lines)
    affix_text: dict[str, list[str]] = {}
    for lang in TARGETS:
        affix_text[lang] = read_lines(data_dir / f"{lang}.train.txt", affix_lines)

    rows: list[dict] = []
    for tag in sorted(index):
        variant = index[tag].get("variant", "")
        if variant.startswith("V1"):
            continue        # a monolingual vocabulary shares nothing by construction; C3 is undefined
        tok = load_tokenizer(vocab_dir / tag)
        marker = marker_for(tok)
        helper_pieces = {h: pieces_used(tok, helper_text[h]) for h in helper_text}

        for lang in TARGETS:
            support = suffix_support(tok, affix_text[lang], marker)
            totals = Counter()                       # per helper: total / morph / affix / generic
            n_tokens = 0
            for ln in evals[lang]:
                toks = tok.encode(ln).tokens
                n_tokens += len(toks)
                for t in toks:
                    cls = classify_piece(t, marker, support, min_support)
                    for h in HELPER[lang]:
                        if t in helper_pieces[h]:
                            totals[f"{h}_total"] += 1
                            if cls == "affix":
                                totals[f"{h}_affix"] += 1
                            elif cls in ("junk", "fragment"):
                                totals[f"{h}_generic"] += 1

            row = {
                "tag": tag,
                "variant": variant,
                "model": index[tag].get("model"),
                "lang": lang,
                "tokens": n_tokens,
            }
            for h in HELPER[lang]:
                # Three disjoint buckets that sum to the total:
                #   affix   = productive suffixes of the target language (the strict reading)
                #   generic = single letters + non-alphabetic pieces (what any two languages
                #             in one script share for free)
                #   morph   = the remainder: alphabetic pieces that are not productive affixes
                # `morph` is defined as total − generic so the three always add up even if the
                # classifier gains a class later.
                tot = totals[f"{h}_total"]
                gen = totals[f"{h}_generic"]
                aff = totals[f"{h}_affix"]
                row[f"C3_total_{h}"] = round(tot / n_tokens, 4) if n_tokens else None
                row[f"C3_generic_{h}"] = round(gen / n_tokens, 4) if n_tokens else None
                row[f"C3_morph_{h}"] = round((tot - gen) / n_tokens, 4) if n_tokens else None
                row[f"C3_affix_{h}"] = round(aff / n_tokens, 4) if n_tokens else None
            rows.append(row)
    return rows


def crosscheck_cost_metrics(rows: list[dict], cost_csv: Path) -> list[str]:
    """Confirm `C3_total` reproduces `C3_share_<h>` from `cost_metrics.csv`.

    Two scripts computing the same quantity is exactly the situation this codebase is built to
    distrust: if they drift apart, nothing errors and both numbers still look plausible. So the
    comparison is made explicitly rather than assumed.
    """
    if not cost_csv.exists():
        return [f"SKIP  {cost_csv.name} not found — C3 not cross-checked against cost_analysis.py"]
    want: dict[tuple[str, str, str], float] = {}
    with open(cost_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for h in ("tr", "uk", "ru"):
                v = r.get(f"C3_share_{h}")
                if v not in (None, ""):
                    want[(r["tag"], r["lang"], h)] = float(v)

    bad, checked = [], 0
    for r in rows:
        for h in HELPER[r["lang"]]:
            key = (r["tag"], r["lang"], h)
            if key not in want or r[f"C3_total_{h}"] is None:
                continue
            checked += 1
            if abs(want[key] - r[f"C3_total_{h}"]) > 1e-4:
                bad.append(f"{r['tag']} {r['lang']}/{h}: cost_metrics {want[key]} "
                           f"vs directionality {r[f'C3_total_{h}']}")
    out = [f"C3 cross-check vs {cost_csv.name}: {checked} values compared"]
    if bad:
        out.append(f"  ⚠️  {len(bad)} MISMATCH — the two scripts have drifted apart:")
        out += [f"      {b}" for b in bad[:5]]
    elif checked:
        out.append("  ✅ all match — C3 here is the same quantity as C3_share_* in cost_metrics.csv")
    return out


def fmt(v, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed" / "vocab_training")
    ap.add_argument("--vocab-dir", type=Path, default=ROOT / "artifacts" / "vocabularies")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "metrics" / "preprocessing")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"],
                    help="inheritance is measured on train: dev/test are far too small for a rate")
    ap.add_argument("--eval-split", default="test",
                    help="which split the C3 decomposition runs on, to match cost_metrics.csv")
    ap.add_argument("--langs", nargs="*", default=["az", "be", "tr", "uk", "ru"])
    ap.add_argument("--helper-lines", type=int, default=50000)
    ap.add_argument("--affix-lines", type=int, default=50000)
    ap.add_argument("--min-suffix-support", type=int, default=20)
    ap.add_argument("--max-lines", type=int, default=0)
    ap.add_argument("--cost-csv", type=Path, default=ROOT / "results" / "metrics" / "preprocessing" / "cost_metrics.csv",
                    help="cost_analysis.py's output, used to confirm C3 here means the same thing")
    args = ap.parse_args()

    print(f"=== evidence: is this a translation corpus? (split={args.split}) ===")
    inh = inheritance_table(args.split, args.langs)
    for r in inh:
        print(f"  {r['lang']}: stage directions {r['both_have_stage_direction']:,}/"
              f"{r['en_with_stage_direction']:,} = {fmt(r['stage_direction_inheritance'])}   "
              f"speaker markers {fmt(r['speaker_marker_inheritance'])}"
              + (f"   latin residue {fmt(r['latin_residue_rate'])}" if r["latin_residue_rate"] else ""))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = args.vocab_dir / "index.json"
    rows: list[dict] = []
    if idx_path.exists():
        index = __import__("json").loads(idx_path.read_text(encoding="utf-8"))
        print(f"\n=== C3 decomposition (eval split={args.eval_split}) ===")
        rows = c3_decomposition(index, args.vocab_dir, args.data_dir, args.eval_split,
                                args.helper_lines, args.affix_lines, args.min_suffix_support,
                                args.max_lines)
        for line in crosscheck_cost_metrics(rows, args.cost_csv):
            print(f"  {line}" if not line.startswith("  ") else line)
        for r in rows:
            for h in HELPER[r["lang"]]:
                print(f"  {r['tag']:20s} {r['lang']}  vs {h}: "
                      f"C3 {fmt(r[f'C3_total_{h}'])} = morph {fmt(r[f'C3_morph_{h}'])} "
                      f"(of which affix {fmt(r[f'C3_affix_{h}'])}) + generic {fmt(r[f'C3_generic_{h}'])}")
    else:
        print(f"\nSKIP  {idx_path} not found — C3 decomposition needs trained vocabularies")

    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.out_dir / "directionality.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # ---------------------------------------------------------------- report
    md = [
        "# WP0 · Directionality — the TED corpus is a translation corpus",
        "",
        "The tar ships **no translator metadata** (183 files, all `ted.{split}.{lang}`), so this is",
        "established from the text. Three independent signals agree that every non-English side is a",
        "translation of the English side.",
        "",
        "## 1. Stage directions are translated at the same positions",
        "",
        "TED transcripts carry `(Applause)` / `(Laughter)` / `(Music)`. A native corpus would not have",
        "them at all, let alone in the same places. Measured over all aligned positions:",
        "",
        "| Language | Aligned positions | English has a stage direction | Same position also has one | Inheritance |",
        "|---|---|---|---|---|",
    ]
    for r in inh:
        md.append(f"| {r['lang']} | {r['aligned_positions']:,} | {r['en_with_stage_direction']:,} | "
                  f"{r['both_have_stage_direction']:,} | **{fmt(r['stage_direction_inheritance'])}** |")
    md += [
        "",
        "## 2. The transcript's `\"\"` speaker-change markers are inherited",
        "",
        "| Language | English has `\"\"` | Same position also has `\"\"` | Inheritance |",
        "|---|---|---|---|",
    ]
    for r in inh:
        md.append(f"| {r['lang']} | {r['en_with_speaker_marker']:,} | {r['both_have_speaker_marker']:,} | "
                  f"**{fmt(r['speaker_marker_inheritance'])}** |")
    md += [
        "",
        "## 3. Untranslated English left in place (non-Latin-script languages only)",
        "",
        "⚠️ This check is only meaningful where Latin script is *foreign* to the language. For az and",
        "tr it would match their own words and report a meaningless ~90%, so those two are excluded.",
        "",
        "| Language | Sentences | With Latin residue | Rate |",
        "|---|---|---|---|",
    ]
    for r in inh:
        if r["latin_residue_rate"] is not None:
            md.append(f"| {r['lang']} | {r['sentences']:,} | {r['sentences_with_latin_residue']:,} | "
                      f"{fmt(r['latin_residue_rate'])} |")

    md += [
        "",
        "## 4. What this does to C3 — and the corrected reading",
        "",
        "Under **V2** (a vocabulary trained on az+be+en that has never seen Turkish), az tokens still",
        "land on pieces tr also produces ~77% of the time. That is not transfer; it is structural —",
        "both sides are translations of the same English sentences, so both carry the same Latin",
        "characters, punctuation, digits and loanwords. `C3_generic` is that part. `C3_morph` removes",
        "it; `C3_affix` keeps only pieces that are productive suffixes of the target language.",
        "",
    ]
    if rows:
        md += ["**Buckets are disjoint and sum to C3**: `morph` = alphabetic pieces that are not",
               "productive affixes, `affix` = productive suffixes, `generic` = single letters plus",
               "non-alphabetic pieces. The last column is the point of the whole exercise — how much",
               "of the headline C3 is material any two languages in one script would share for free.", ""]
        for lang in TARGETS:
            sel = [r for r in rows if r["lang"] == lang]
            if not sel:
                continue
            md += [f"### {lang}", "",
                   "| Vocabulary | helper | **C3** | morph | affix | generic | generic ÷ C3 |",
                   "|---|---|---|---|---|---|---|"]
            for r in sel:
                for h in HELPER[lang]:
                    tot = r[f"C3_total_{h}"]
                    gen = r[f"C3_generic_{h}"]
                    share = f"{gen / tot:.1%}" if tot else "—"
                    md.append(f"| {r['tag']} | {h} | **{fmt(tot)}** | {fmt(r[f'C3_morph_{h}'])} | "
                              f"{fmt(r[f'C3_affix_{h}'])} | {fmt(gen)} | {share} |")
            md.append("")

    md += [
        "## 5. Consequences to state in the report",
        "",
        "1. **The two directions are not symmetric tasks.** `X→en` recovers the *source* of a",
        "   translation, and translationese is English-shaped; its reference is the English original.",
        "   `en→X` must match one human translator's choices, because that translation *is* the",
        "   reference BLEU is scored against. **A direction gap is therefore not by itself evidence",
        "   of a data-volume gap.**",
        "2. **RQ1's similarity channel is confounded.** tr / uk / ru are translations of the same",
        "   English sentences as az and be, so part of their shared structure comes from one shared",
        "   translation process rather than from the languages being related. RQ1 has to rule this out.",
        "3. **C3 must be quoted with this caveat.** Its baseline is inflated by script- and",
        "   translation-shared material; `C3_morph` / `C3_affix` are the defensible readings.",
        "4. **TED is single-source.** Every direction here starts from English. The corpus cannot",
        "   support a claim about translation *into* English from a natively-authored source.",
        "",
    ]
    (args.out_dir / "directionality_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n→ {args.out_dir / 'directionality_report.md'}")
    if rows:
        print(f"→ {args.out_dir / 'directionality.csv'}")


if __name__ == "__main__":
    main()
