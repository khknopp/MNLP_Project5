#!/usr/bin/env python3
"""
WP0 · Extract plain text for 6 languages from TED.tar, writing it out by train/dev/test split.

Why unpack it ourselves instead of using load_dataset:
  `davidstap/ted_talks` is a script-based dataset (runs arbitrary python), and the HF
  datasets-server does not do parquet conversion; also the script's config "zh" is
  sparse Cantonese, so config names are an extremely easy trap to fall into.
  Reading the text straight from the tar is the most controllable option.

Usage: python prep_data.py
Output: data/processed/vocab_training/{lang}.{train,dev,test}.txt  +  data/processed/vocab_training/SIZES.md
"""
from __future__ import annotations

import tarfile
from pathlib import Path

LANGS = ["en", "az", "be", "tr", "uk", "ru"]
SPLITS = {"train": "train", "dev": "validation", "test": "test"}

# Known baselines (previously measured on miao on this machine; used to self-check whether extraction read the wrong thing)
EXPECT = {"az": {"train": 5946, "dev": 671, "test": 903},
          "be": {"train": 4509, "dev": 248, "test": 664},
          "en": {"train": 258098}}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    tar_path = root / "data" / "TED.tar"
    out = root / "data" / "processed" / "vocab_training"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, int, str]] = []
    with tarfile.open(tar_path) as tf:
        names = set(tf.getnames())
        for lang in LANGS:
            for split, folder in SPLITS.items():
                member = f"{folder}/ted.{split}.{lang}"
                if member not in names:
                    print(f"  ⚠️  {member} not in tar, skipping")
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                lines = [ln.strip() for ln in f.read().decode("utf-8", "replace").splitlines()]
                lines = [ln for ln in lines if ln]          # drop blank lines — matches the convention used before
                (out / f"{lang}.{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                rows.append((lang, split, len(lines), member))
                print(f"  {lang}.{split:5s} {len(lines):>8,} lines")

    # ---- Self-check: do these line counts match the earlier measurements?
    md = ["# Corpus sizes (unpacked from TED.tar)", "", "| lang | train | dev | test |", "|---|---|---|---|"]
    by = {(l, s): n for l, s, n, _ in rows}
    for lang in LANGS:
        md.append(f"| {lang} | {by.get((lang,'train'),0):,} | {by.get((lang,'dev'),0):,} "
                  f"| {by.get((lang,'test'),0):,} |")
    md += ["", "## Self-check", ""]
    ok = True
    for lang, sp, exp in [(l, s, n) for l, d in EXPECT.items() for s, n in d.items()]:
        got = by.get((lang, sp))
        flag = "✅" if got == exp else "❌"
        if got != exp:
            ok = False
        md.append(f"- {flag} {lang}.{sp}: unpacked {got:,} / baseline {exp:,}")
    md += ["", "All matched ✅" if ok else "**Mismatch — the extraction convention needs to be checked**"]
    (out / "SIZES.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    print("\n" + "\n".join(md[md.index("## Self-check"):]))
    print(f"\n→ {out}/SIZES.md")


if __name__ == "__main__":
    main()
