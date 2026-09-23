#!/usr/bin/env python3
"""Check local artifacts against manifest.json — proof that the team shares one dataset.

The corpus cannot be committed (CC-BY-NC-ND-4.0, see .gitignore), so everyone
downloads TED.tar and re-runs the pipeline themselves. "Are we on the same data?"
therefore cannot be answered by eyeballing files. This does it: manifest.json
records the line count and the first 16 hex digits of the sha256 of every
artifact; recompute locally and compare.

If anyone changes the split / dedup / filtering strategy without updating the
manifest, this goes red immediately.

Usage:
    python verify.py                          # check data/processed/parallel/
    python verify.py --dir data/processed/parallel
    python verify.py --manifest data/processed/parallel/manifest.json --dir data/processed/parallel
    python verify.py --vocabs                 # check artifacts/vocabularies/ against artifacts/vocabularies/index.json

Exit code: 0 = all match; 1 = mismatch (every offending file is listed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_16(p: Path) -> str:
    """Must match align.py's _sha() exactly: chunked read + first 16 hex digits."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def count_lines(p: Path) -> int:
    """Count lines the way align.py writes them: a trailing newline is not a line."""
    n = 0
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            n += chunk.count(b"\n")
    if p.stat().st_size and p.read_bytes()[-1:] != b"\n":
        n += 1
    return n


def verify_vocabs(index_path: Path, base: Path, quiet: bool) -> int:
    """Check every vocabulary against the hash recorded in `artifacts/vocabularies/index.json`.

    This is a *different* kind of check from the parallel-corpus one, and both are needed:
    that one is about the **data**, this one is about the **segmentation**. Two people can
    share one corpus byte for byte and still segment it differently — a different `tokenizers`
    version, a different training-text truncation, an unnoticed retrain — and then their
    C1–C5 metrics and their model inputs silently stop being comparable, with no error
    anywhere. Identical hash ⇒ byte-identical `tokenizer.json` ⇒ identical segmentation.

    The vocabularies themselves are corpus derivatives and are never committed (CC-BY-NC-ND),
    which is exactly why `index.json` — line counts and hashes only, no text — **is** committed.
    """
    if not index_path.exists():
        print(f"FAIL  {index_path} not found")
        print("      Run `python train_vocabs.py --vocab-size 8000` (or `--rehash`) first.")
        return 1

    index = json.loads(index_path.read_text(encoding="utf-8"))
    print(f"vocab index: {index_path}")
    print(f"  artifacts : {base}")
    print(f"  to check  : {len(index)} vocabularies\n")

    bad: list[tuple[str, str]] = []
    missing: list[str] = []
    unhashed: list[str] = []

    for tag, meta in sorted(index.items()):
        p = base / tag / "tokenizer.json"
        want = meta.get("tokenizer_sha256_16")
        if want is None:
            unhashed.append(tag)
            continue
        if not p.exists():
            missing.append(tag)
            continue
        got = sha256_16(p)
        if got != want:
            bad.append((tag, f"sha256 expected {want} got {got}"))
        elif not quiet:
            size = meta.get("vocab_size_actual", "?")
            print(f"  OK   {tag:26s} {size:>6} pieces  {got}")

    print()
    if missing:
        print(f"FAIL  {len(missing)} vocabular(y/ies) not on disk:")
        for n in missing[:10]:
            print(f"        {n}")
    if unhashed:
        print(f"FAIL  {len(unhashed)} entr(y/ies) in the index carry no hash, so nothing can be checked:")
        for n in unhashed[:10]:
            print(f"        {n}")
        print("      Run `python train_vocabs.py --rehash` to record them from the artifacts on disk.")
    if bad:
        print(f"FAIL  {len(bad)} vocabular(y/ies) differ from the index:")
        for n, why in bad[:10]:
            print(f"        {n}: {why}")

    if missing or bad or unhashed:
        print("\n-> Your segmentation is NOT the same as this index.")
        print("   Do not run downstream experiments: your model inputs, and therefore your BLEU,")
        print("   would not be comparable with the rest of the team's.")
        return 1

    print(f"OK    all {len(index)} vocabularies match the index — you and the team segment identically.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "processed" / "parallel" / "manifest.json")
    ap.add_argument("--dir", type=Path, default=None, help="artifact directory (defaults to the manifest's own directory)")
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    ap.add_argument("--vocabs", action="store_true",
                    help="check artifacts/vocabularies/ against artifacts/vocabularies/index.json instead of the parallel corpora")
    ap.add_argument("--vocab-index", type=Path, default=ROOT / "artifacts" / "vocabularies" / "index.json")
    args = ap.parse_args()

    if args.vocabs:
        return verify_vocabs(args.vocab_index, args.vocab_index.parent, args.quiet)

    if not args.manifest.exists():
        print(f"FAIL  {args.manifest} not found")
        print("      Run `python align.py emit` first to produce the artifacts and the manifest.")
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    base = args.dir or args.manifest.parent

    print(f"manifest: {args.manifest}")
    print(f"  source    : {manifest.get('source', '?')}")
    print(f"  alignment : {manifest.get('alignment', '?')}")
    print(f"  generated : {manifest.get('generated_at', '?')}")
    print(f"  artifacts : {base}")
    print(f"  to check  : {len(files)} files\n")

    bad: list[tuple[str, str]] = []
    missing: list[str] = []

    for name, meta in sorted(files.items()):
        p = base / name
        if not p.exists():
            missing.append(name)
            continue
        got_n = count_lines(p)
        got_h = sha256_16(p)
        if got_n != meta["lines"] or got_h != meta["sha256_16"]:
            why = []
            if got_n != meta["lines"]:
                why.append(f"lines expected {meta['lines']:,} got {got_n:,}")
            if got_h != meta["sha256_16"]:
                why.append(f"sha256 expected {meta['sha256_16']} got {got_h}")
            bad.append((name, "; ".join(why)))
        elif not args.quiet:
            print(f"  OK   {name:28s} {got_n:>9,} lines  {got_h}")

    print()
    if missing:
        print(f"FAIL  {len(missing)} file(s) missing:")
        for n in missing[:10]:
            print(f"        {n}")
        if len(missing) > 10:
            print(f"        ... and {len(missing) - 10} more")
    if bad:
        print(f"FAIL  {len(bad)} file(s) differ:")
        for n, why in bad[:20]:
            print(f"        {n}: {why}")
        if len(bad) > 20:
            print(f"        ... and {len(bad) - 20} more")

    if missing or bad:
        print("\n-> Your data is NOT the same as this manifest.")
        print("   Do not run downstream experiments: your BLEU would not be comparable.")
        print("   Align on a version with whoever produced this manifest first.")
        return 1

    print(f"OK    all {len(files)} files match the manifest — you and the team share one dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
