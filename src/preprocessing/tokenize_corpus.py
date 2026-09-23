#!/usr/bin/env python3
"""Segment the corpus with the WP0 vocabularies — downstream can consume it directly.

**Why this step exists.** `out/vocabs/*/tokenizer.json` only ships the vocabulary;
whoever picks it up still has to decide which pre-tokenizer to use, whether to
prepend, where the special tokens go. Get any of that wrong and your segmentation
no longer matches the WP0 cost metrics — **and it fails silently**, quietly making
BLEU scores incomparable. So the tokenized corpus ships as an artifact too.

Output (one directory per vocabulary variant):
    out/tokenized/{tag}/{lang}.{split}.tok      space-separated pieces (fairseq convention)
    out/tokenized/{tag}/{lang}.{split}.ids      ⟨optional, --ids⟩ token ids
    out/tokenized/manifest.json                 line counts + sha256 + vocab tag + settings

WARNING: the output is a derivative of the corpus (CC-BY-NC-ND-4.0) and is
**also never committed** — everyone generates it locally.

Usage:
    python tokenize_corpus.py                        # V3 unigram only (the release variant)
    python tokenize_corpus.py --variants V3_8000_bpe V4_8000_unigram_a0.2
    python tokenize_corpus.py --langs az be --splits train
    python tokenize_corpus.py --ids
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

LANGS = ["en", "az", "be", "tr", "uk", "ru"]
SPLITS = ["train", "dev", "test"]


def sha256_16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def tokenize_file(tok: Tokenizer, src: Path, dst_tok: Path, dst_ids: Path | None) -> tuple[int, int]:
    """Return (lines, total pieces). Blank lines are preserved — position indices must not shift."""
    lines = src.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    enc = tok.encode_batch(lines, add_special_tokens=False)
    pieces = [" ".join(e.tokens) for e in enc]
    dst_tok.write_text("\n".join(pieces) + "\n", encoding="utf-8")
    if dst_ids is not None:
        dst_ids.write_text("\n".join(" ".join(map(str, e.ids)) for e in enc) + "\n", encoding="utf-8")
    return len(pieces), sum(len(e.tokens) for e in enc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--vocab-dir", type=Path, default=ROOT / "out" / "vocabs")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / "tokenized")
    ap.add_argument("--variants", nargs="*", default=["V3_8000_unigram"],
                    help="release variant V3_8000_unigram by default; use others only for ablations")
    ap.add_argument("--langs", nargs="*", default=LANGS)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--ids", action="store_true", help="also emit token id files")
    args = ap.parse_args()

    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "space-separated pieces (fairseq convention). .ids lines correspond 1:1 to .tok.",
        "variants": {},
    }

    for tag in args.variants:
        vdir = args.vocab_dir / tag
        if not (vdir / "tokenizer.json").exists():
            print(f"SKIP  {tag}: no {vdir}/tokenizer.json (run train_vocabs.py first)")
            continue

        tok = Tokenizer.from_file(str(vdir / "tokenizer.json"))
        out = args.out_dir / tag
        out.mkdir(parents=True, exist_ok=True)
        entry: dict = {"vocab_size": tok.get_vocab_size(), "files": {}}
        print(f"\n== {tag} ==  vocab {tok.get_vocab_size():,}")

        for split in args.splits:
            for lang in args.langs:
                src = args.data_dir / f"{lang}.{split}.txt"
                if not src.exists():
                    continue
                dst_tok = out / f"{lang}.{split}.tok"
                dst_ids = out / f"{lang}.{split}.ids" if args.ids else None
                n_lines, n_tokens = tokenize_file(tok, src, dst_tok, dst_ids)

                # Self-check: the segmented line count must equal the input's exactly,
                # otherwise every position index downstream is wrong.
                if n_lines == 0:
                    dst_tok.unlink(missing_ok=True)
                    continue
                got = len(dst_tok.read_text(encoding="utf-8").rstrip("\n").split("\n"))
                if got != n_lines:
                    raise SystemExit(f"FAIL {tag}/{lang}.{split}: {n_lines} lines -> {got}, alignment broken")

                entry["files"][dst_tok.name] = {"lines": n_lines, "tokens": n_tokens,
                                                "sha256_16": sha256_16(dst_tok)}
                if dst_ids is not None:
                    entry["files"][dst_ids.name] = {"lines": n_lines, "sha256_16": sha256_16(dst_ids)}
                fert = n_tokens / max(n_lines, 1)
                print(f"  {lang}.{split:5s} {n_lines:>9,} lines  {n_tokens:>12,} pieces  avg {fert:>7.1f} pieces/line")

        manifest["variants"][tag] = entry

    if not manifest["variants"]:
        raise SystemExit("No variant was processed — check --variants and out/vocabs/")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n-> {args.out_dir}/manifest.json")
    print("WARNING: derivative of the corpus — do not commit (CC-BY-NC-ND-4.0)")


if __name__ == "__main__":
    main()
