#!/usr/bin/env python3
"""
WP0 · Alignment contract and parallel-corpus delivery

**This is the one data entry point for the whole team.** Design intent: make
"accidentally getting the line numbers wrong" impossible to do.

Core facts (measured on this machine):
    Every language has **strictly equal** raw line counts on every split
    train 258,098 · dev 6,049 · test 7,213
    An empty line = an alignment placeholder for "this position has no
    translation in this language", **not dirty data**.

So the public surface **exposes only "fetch by position"**, never "read the file
line by line" — there is no notion of a "line" in the interface, so callers never
get a chance to drop empty lines.

Command line:
    python align.py extract                       # tar → data/raw/ (empty lines preserved)
    python align.py counts                        # real sentence-pair counts per language pair
    python align.py check                         # leakage / duplicate checks
    python align.py emit --pairs az-en be-en --out out/parallel
    python align.py pivot --out out/pivot         # az-tr-en / be-uk-en / be-ru-en triples
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

# This file lives in preprocessing/; data/ and out/ live at the repository root,
# so that every script resolves them the same way no matter where it is invoked from.
ROOT = Path(__file__).resolve().parent.parent
TAR = ROOT / "data" / "TED.tar"
RAW = ROOT / "data" / "raw"

# directory name inside the tar → the split name we expose
SPLIT_DIR = {"train": "train", "dev": "validation", "test": "test"}

# The 6 languages this project uses. extract can pull all 59 if needed, but by default it takes
# only these 6 (saves disk/time)
PROJECT_LANGS = ["en", "az", "be", "tr", "uk", "ru"]

# The language pairs the team has to deliver (grouped by purpose, see notebooks/data_overview.ipynb §3)
DELIVER_PAIRS = [
    ("az", "en"), ("be", "en"),          # main line
    ("tr", "en"), ("uk", "en"), ("ru", "en"),   # training data for the helper
    ("be", "uk"), ("be", "ru"),          # direct comparison for RQ1
    ("az", "tr"),                        # helper channel for az
    ("az", "be"),                        # only 310 sentences, pivot material
]

PIVOTS = [
    ("az", "tr", "en"), ("be", "uk", "en"), ("be", "ru", "en"),
]


# --------------------------------------------------------------- reading raw files
def _iter_tar_lang(lang: str):
    """Pull this language's raw lines for all three splits out of the tar (**empty lines kept**)."""
    out: dict[str, list[str]] = {}
    with tarfile.open(TAR) as tf:
        names = set(tf.getnames())
        for split, folder in SPLIT_DIR.items():
            member = f"{folder}/ted.{split}.{lang}"
            if member not in names:
                raise KeyError(f"{member} is not in the tar")
            data = tf.extractfile(member).read().decode("utf-8", "replace")
            lines = data.split("\n")
            if lines and lines[-1] == "":
                lines = lines[:-1]      # drop only the trailing newline at EOF; keep every empty line in between
            out[split] = lines
    return out


def extract(langs: list[str] = PROJECT_LANGS, force: bool = False) -> None:
    """Unpack the tar into data/raw/{split}/{lang}.txt — **empty lines kept verbatim**."""
    for lang in langs:
        per_split = _iter_tar_lang(lang)
        for split, lines in per_split.items():
            d = RAW / split
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{lang}.txt"
            if p.exists() and not force:
                print(f"  already exists, skipping {p} (use --force to overwrite)")
                continue
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            nonempty = sum(1 for x in lines if x.strip())
            print(f"  {split}/{lang}.txt  raw {len(lines):>9,}  non-empty {nonempty:>9,}")


def _raw(split: str, lang: str) -> list[str]:
    p = RAW / split / f"{lang}.txt"
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python align.py extract` first")
    lines = p.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


# ----------------------------------------------- public interface (exposes "position" only)
def positions(split: str, langs: list[str]) -> list[int]:
    """Position numbers where **all** the given languages are non-empty."""
    cols = [_raw(split, l) for l in langs]
    n = len(cols[0])
    for l, c in zip(langs, cols):
        if len(c) != n:
            raise SystemExit(
                f"❌ alignment broken: under {split}, {langs[0]}={n} lines, {l}={len(c)} lines."
                " This should never happen — it means someone dropped empty lines, re-run `align.py extract --force`"
            )
    return [i for i in range(n) if all(c[i].strip() for c in cols)]


def pairs(split: str, src: str, tgt: str):
    """Yield (position number, src text, tgt text). The position number is kept so this pair can be stitched with other pairs for pivot."""
    s, t = _raw(split, src), _raw(split, tgt)
    for i in positions(split, [src, tgt]):
        yield i, s[i].strip(), t[i].strip()


def triples(split: str, langs: list[str]):
    """Triples (for pivot): all three columns are non-empty at the same position number."""
    cols = {l: _raw(split, l) for l in langs}
    for i in positions(split, langs):
        yield (i, *(cols[l][i].strip() for l in langs))


def pair_counts(split: str, lang_pairs=None) -> dict[str, int]:
    lang_pairs = lang_pairs or DELIVER_PAIRS
    return {f"{a}-{b}": len(positions(split, [a, b])) for a, b in lang_pairs}


# ------------------------------------------------------------------ writing artifacts
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def emit(pairs_to_emit, splits, out_dir: Path) -> None:
    """Write the parallel corpus + manifest to disk. The manifest is what makes the whole team's BLEU comparable, don't skip it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "data/TED.tar (davidstap/ted_talks, CC-BY-NC-ND-4.0)",
        "alignment": "Strictly aligned by line number, empty lines are placeholders; this artifact contains only positions where both columns are non-empty",
        "files": {},
    }
    for split in splits:
        for a, b in pairs_to_emit:
            rows = [(i, s, t) for i, s, t in pairs(split, a, b)]
            if not rows:
                print(f"  skipping {a}-{b}.{split} (0 sentences)")
                continue
            stem = f"{a}-{b}.{split}"
            for col, name in ((1, a), (2, b)):
                p = out_dir / f"{stem}.{name}"
                p.write_text("\n".join(r[col] for r in rows) + "\n", encoding="utf-8")
                manifest["files"][p.name] = {"lines": len(rows), "sha256_16": _sha(p)}
            # store the position numbers separately: pivot / three-way stitching needs them
            pi = out_dir / f"{stem}.pos"
            pi.write_text("\n".join(str(r[0]) for r in rows) + "\n", encoding="utf-8")
            manifest["files"][pi.name] = {"lines": len(rows), "sha256_16": _sha(pi)}
            print(f"  {stem}: {len(rows):,} sentences")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n→ {out_dir}/manifest.json")


def emit_pivots(out_dir: Path, splits=("train", "dev", "test")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for langs in PIVOTS:
            rows = list(triples(split, list(langs)))
            if not rows:
                print(f"  skipping {'-'.join(langs)}.{split} (0 sentences)")
                continue
            p = out_dir / f"{'-'.join(langs)}.{split}.tsv"
            with open(p, "w", encoding="utf-8") as f:
                f.write("pos\t" + "\t".join(langs) + "\n")
                for r in rows:
                    f.write("\t".join(str(x) for x in r) + "\n")
            print(f"  {'-'.join(langs)}.{split}: {len(rows):,} sentences")


# ------------------------------------------------------------------ leakage checks
def check(splits=("train", "dev", "test")) -> None:
    """Leakage and duplicate checks. The conclusions have to go into the report's Limitations — otherwise all four people's BLEU carries bias."""
    print("=== alignment completeness ===")
    for split in splits:
        cols = {l: len(_raw(split, l)) for l in PROJECT_LANGS}
        ok = len(set(cols.values())) == 1
        print(f"  {split}: " + (" all %d lines ✅" % next(iter(cols.values())) if ok else f"❌ {cols}"))

    print("\n=== sentence-pair counts per language pair ===")
    header = f"{'pair':10s}" + "".join(f"{s:>10s}" for s in splits)
    print(header)
    for a, b in DELIVER_PAIRS:
        print(f"{a}-{b:6s}" + "".join(f"{len(positions(s,[a,b])):>10,}" for s in splits))

    print("\n=== leakage check (duplicate English sentences) ===")
    tr = [x.strip() for x in _raw("train", "en") if x.strip()]
    trs = set(tr)
    print(f"  duplicate English sentences inside train: {len(tr)-len(trs):,} / {len(tr):,}")
    for sp in ("dev", "test"):
        s = set(x.strip() for x in _raw(sp, "en") if x.strip())
        print(f"  train ∩ {sp}: {len(trs & s):,} sentences ({len(trs & s)/max(len(s),1)*100:.2f}% of {sp})")
    for lg in ("az", "be"):
        hits = tot = 0
        for _, src, tgt in pairs("test", lg, "en"):
            tot += 1
            if tgt in trs:
                hits += 1
        print(f"  {lg} test: share of English sentences that also appear in train: {hits}/{tot} ({hits/max(tot,1)*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="tar → data/raw/ (empty lines preserved)")
    e.add_argument("--langs", nargs="*", default=PROJECT_LANGS)
    e.add_argument("--force", action="store_true")

    sub.add_parser("counts", help="print sentence-pair counts for each language pair")
    sub.add_parser("check", help="alignment completeness + leakage checks")

    m = sub.add_parser("emit", help="write the parallel corpus + manifest to disk")
    m.add_argument("--pairs", nargs="*", default=[f"{a}-{b}" for a, b in DELIVER_PAIRS])
    m.add_argument("--splits", nargs="*", default=["train", "dev", "test"])
    m.add_argument("--out", type=Path, default=ROOT / "out" / "parallel")

    p = sub.add_parser("pivot", help="write the pivot triples to disk")
    p.add_argument("--out", type=Path, default=ROOT / "out" / "pivot")

    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.langs, args.force)
    elif args.cmd == "counts":
        for split in SPLIT_DIR:
            print(f"--- {split}")
            for k, v in pair_counts(split).items():
                print(f"    {k:8s} {v:>10,}")
    elif args.cmd == "check":
        check()
    elif args.cmd == "emit":
        pairs_to_emit = [tuple(s.split("-")) for s in args.pairs]
        emit(pairs_to_emit, args.splits, args.out)
    elif args.cmd == "pivot":
        emit_pivots(args.out)


if __name__ == "__main__":
    main()
