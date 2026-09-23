#!/usr/bin/env python3
"""
WP0 · Train the V1–V5 subword vocabularies.

Design principle: **the only thing that varies is which languages' text the vocab is trained on**.
Vocab size, model type, special tokens and iteration counts are all fixed —— otherwise Δ can't be attributed.

Usage:
    python train_vocabs.py --vocab-size 8000                 # runs V1/V2/V3
    python train_vocabs.py --vocab-size 8000 --variants V4    # only the temperature-sampling variant
    python train_vocabs.py --vocab-size 8000 --alphas 0.2 0.5 0.8
    python train_vocabs.py --rehash                          # no training: (re)record the hashes of existing artifacts

Output: artifacts/vocabularies/<variant>_<size>[_<suffix>]/tokenizer.json
      plus artifacts/vocabularies/index.json (records the exact construction args of every variant, for cost_analysis to reproduce)

**Why every entry also carries `tokenizer_sha256_16`.** `manifest.json` / `verify.py` hash only the
parallel corpora, which never touch a tokenizer — so vocabulary training and the C1–C5 numbers are
covered by no hash at all. If two people train "V3" on slightly different inputs, or with different
`tokenizers` versions, they get **different segmentations and comparable-looking numbers, with no error
anywhere**. The hash is what turns `verify.py --vocabs` into a real check: identical hash ⇒ byte-identical
tokenizer ⇒ identical segmentation. `--rehash` backfills hashes for artifacts trained before this existed,
without retraining anything (hash-only, changes no vocabulary).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

LANGS = ["en", "az", "be", "tr", "uk", "ru"]

# which languages' text each variant's vocab is trained on
VARIANT_LANGS: dict[str, list[str]] = {
    "V1_az": ["az"],                    # monolingual upper bound (the ceiling of az representation)
    "V1_be": ["be"],
    "V2": ["az", "be", "en"],           # low-resource only + hub
    "V3": ["az", "be", "tr", "uk", "ru", "en"],   # plus the helper languages (main experiment)
    "V5": ["az", "be", "tr", "uk", "ru", "en"],   # joint training after unified transliteration (recovery R2)
}

# ---------------------------------------------------------------- transliteration table
# ⚠️ This is a **linguistic judgment**, not an implementation detail —— confirm it with the group/advisor before submitting.
# Motivation: the advisor brought up Urdu/Hindi (same language, different writing systems),
#       "if you could convert them both into the same script, it probably would help
#        knowledge transfer, because then basically the same strings are used".
# Here we only do a **deterministic, character-by-character** mapping, with no context-dependent linguistic processing.
# The goal is not to spell az as proper Turkish, but to align "cognate affixes" at the string level.
TRANSLIT: dict[str, dict[str, str]] = {
    # the three letters in az's Latin that tr lacks; mapped to the closest-sounding counterpart in tr
    "az": {"ə": "e", "x": "h", "q": "k"},
    # the letters in be's Cyrillic that ru lacks
    "be": {"ў": "у", "і": "и", "э": "е"},
    # the letters in uk's Cyrillic that ru lacks
    "uk": {"і": "и", "ї": "и", "є": "е", "ґ": "г"},
}


def sha256_16(p: Path) -> str:
    """Must match align.py's _sha() and verify.py's sha256_16(): chunked read, first 16 hex digits."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def translit(text: str, lang: str) -> str:
    table = TRANSLIT.get(lang)
    if not table:
        return text
    for src, dst in table.items():
        text = text.replace(src, dst)
    return text


# ------------------------------------------------------- temperature sampling (recovery R1)
def temperature_mix(
    corpora: dict[str, list[str]], alpha: float, budget: int, rng: random.Random
) -> list[str]:
    """Resample each language's lines with p_i ∝ n_i^α and concatenate them into one training text.

    α = 1 → natural proportions (equivalent to V3)
    α < 1 → low-resource languages get upsampled (scarce languages get a relatively larger share)
    """
    sizes = {k: len(v) for k, v in corpora.items()}
    total = sum(sizes.values())
    weights = {k: (n ** alpha) for k, n in sizes.items()}
    z = sum(weights.values())
    out: list[str] = []
    for lang, w in weights.items():
        want = int(round(budget * w / z))
        pool = corpora[lang]
        if not pool or want <= 0:
            continue
        # sampling with replacement is allowed: when upsampling, want can exceed that language's own line count
        out.extend(rng.choice(pool) for _ in range(want))
    rng.shuffle(out)
    return out


def build_tokenizer(model_type: str, vocab_size: int) -> tuple[Tokenizer, trainers.Trainer]:
    special = ["<s>", "</s>", "<unk>", "<pad>"]
    if model_type == "unigram":
        tok = Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        # ⚠️ Measured pitfall (2026-09-15): `special_tokens=["<unk>"]` only puts <unk> into the vocab,
        # it does **not** bind it as the fallback target —— you must separately pass `unk_token="<unk>"`, otherwise
        # the serialized unk_id is None, and encoding any out-of-vocab character throws
        # "Encountered an unknown token but `unk_id` is missing".
        # (`byte_fallback` is silently ignored in this version of tokenizers, don't use it.)
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special,
            unk_token="<unk>",
            show_progress=False,
        )
    else:
        # SentencePiece-style BPE: Metaspace pre-tokenization + merge-based BPE.
        # ⚠️ Deliberately **not** byte-level (ByteLevel): byte-level splits non-ASCII characters by byte
        # (az's ə is 2 bytes, Cyrillic letters are 2 bytes), so fertility is inherently higher,
        # and it is no longer the same experimental condition as Unigram+Metaspace —— what you'd be comparing then
        # is "a different segmentation scheme", not "a different merge algorithm". Use Metaspace on both sides,
        # and BPE vs Unigram becomes a single-variable comparison.
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special,
            unk_token="<unk>",
            show_progress=False,
        )
    return tok, trainer


def selfcheck(tok: Tokenizer, tag: str) -> None:
    """Self-check before delivery: no variant's vocab may throw on out-of-vocab text.

    This check is necessary, not fussiness —— when a teammate trains a model on V3 and the corpus
    picks up a name from a non-6-language or an Arabic word, it crashes mid-epoch, and it is very hard to track down.
    """
    import json as _json
    d = _json.loads(tok.to_str())
    unk_id = d["model"].get("unk_id")
    probe = "Привет мир שלום 日本語 ＡＢＣ"
    try:
        n = len(tok.encode(probe).tokens)
        print(f"   self-check {tag}: unk_id={unk_id} out-of-vocab text is encodable ({n} pieces) ✅")
    except Exception as e:
        raise SystemExit(f"   ❌ {tag} self-check failed: {type(e).__name__}: {e}")


def rehash(out_dir: Path, index: dict[str, dict], idx_path: Path) -> None:
    """Hash-only pass: record `tokenizer_sha256_16` for artifacts that predate the field.

    Trains nothing and changes no vocabulary. A `tokenizer.json` already on disk has a hash
    as a matter of fact — this just writes it down, so that `verify.py --vocabs` has something
    to check a teammate's copy against. Run it once on an existing `artifacts/vocabularies/`.
    """
    recorded = already = 0
    for tag in sorted(index):
        tj = out_dir / tag / "tokenizer.json"
        if not tj.exists():
            print(f"  MISSING  {tag}/tokenizer.json")
            continue
        digest = sha256_16(tj)
        prev = index[tag].get("tokenizer_sha256_16")
        if prev is None:
            recorded += 1
            print(f"  +        {tag:26s} {digest}")
        elif prev != digest:
            recorded += 1
            print(f"  ⚠️       {tag:26s} recorded {prev}, file is {digest} — artifact changed since hashing")
        else:
            already += 1
        index[tag]["tokenizer_sha256_16"] = digest

    orphans = sorted(
        d.name for d in out_dir.iterdir()
        if d.is_dir() and (d / "tokenizer.json").exists() and d.name not in index
    )
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{recorded} recorded, {already} already correct → {idx_path}")
    if orphans:
        print(f"⚠️  {len(orphans)} vocab dir(s) present but absent from index.json "
              f"(cost_analysis.py cannot use them):")
        for o in orphans[:10]:
            print(f"      {o}")


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--data-dir", type=Path, default=root / "data" / "processed" / "vocab_training")
    ap.add_argument("--out-dir", type=Path, default=root / "artifacts" / "vocabularies")
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--model", choices=["unigram", "bpe"], default="unigram")
    ap.add_argument("--variants", nargs="*", default=["V1_az", "V1_be", "V2", "V3", "V5", "V4"],
                    help="which variants to train; V4 means the temperature-sampling α sweep (used with --alphas)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.2, 0.5, 0.8],
                    help="the α sweep for V4; α=1 is equivalent to V3, no need to run it")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-lines-per-lang", type=int, default=0,
                    help="max lines per language used to train the vocab (0 = use all). "
                         "Large languages dominate training time; truncation only affects the vocab, not the parallel corpus")
    ap.add_argument("--rehash", action="store_true",
                    help="train nothing: only (re)record tokenizer_sha256_16 for the artifacts already in --out-dir")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ⚠️ Merge rather than overwrite —— otherwise running BPE wipes out all the existing Unigram variant entries
    idx_path = args.out_dir / "index.json"
    index: dict[str, dict] = (
        json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    )

    # --rehash is hash-only and must not require the corpus, so it runs before anything is read
    if args.rehash:
        rehash(args.out_dir, index, idx_path)
        return

    # Read in each language's training text (read train only, never dev/test —— that would contaminate downstream evaluation)
    corpora: dict[str, list[str]] = {}
    for lang in LANGS:
        p = args.data_dir / f"{lang}.train.txt"
        if not p.exists():
            raise SystemExit(f"missing file {p} —— run prep_data.py first")
        corpora[lang] = read_lines(p)
        if args.max_lines_per_lang:
            corpora[lang] = corpora[lang][: args.max_lines_per_lang]
    print("corpus lines: " + "  ".join(f"{k}={len(v):,}" for k, v in corpora.items()))

    jobs: list[tuple[str, str, list[str]]] = []  # (variant name, suffix, training text)

    for v in args.variants:
        if v == "V4":
            continue        # V4 is not a single language combination; expanded separately below as an α sweep
        langs = VARIANT_LANGS[v]
        text = []
        for lang in langs:
            ls = corpora[lang]
            text.extend(translit(x, lang) if v == "V5" else x for x in ls)
        jobs.append((v, "", text))

    # V4: temperature sampling. ⚠️ Must be governed by --variants —— earlier this appended unconditionally,
    # so --variants V3 would also train all three V4 variants along the way, wasting a dozen minutes for nothing.
    if "V4" in args.variants:
        budget = sum(len(corpora[l]) for l in VARIANT_LANGS["V3"])
        for a in args.alphas:
            text = temperature_mix({l: corpora[l] for l in VARIANT_LANGS["V3"]}, a, budget, rng)
            jobs.append(("V4", f"a{a}", text))

    for name, suffix, text in jobs:
        # tag must carry the model name: otherwise running BPE silently overwrites the Unigram artifact of the same name
        tag = f"{name}_{args.vocab_size}_{args.model}" + (f"_{suffix}" if suffix else "")
        print(f"\n== {tag} ==  training lines {len(text):,} / vocab {args.vocab_size}")
        tok, trainer = build_tokenizer(args.model, args.vocab_size)
        tok.train_from_iterator(text, trainer)
        selfcheck(tok, tag)
        d = args.out_dir / tag
        d.mkdir(parents=True, exist_ok=True)
        tok_path = d / "tokenizer.json"
        tok.save(str(tok_path))
        actual = tok.get_vocab_size()
        digest = sha256_16(tok_path)
        # If this tag already existed with a *different* hash, the numbers everywhere downstream
        # just changed. Say so loudly — a silent vocabulary change is the exact failure mode
        # this field exists to prevent.
        prev = index.get(tag, {}).get("tokenizer_sha256_16")
        if prev and prev != digest:
            print(f"   ⚠️  {tag} was retrained: sha256 {prev} → {digest} — every C1–C5 number for it changed")
        index[tag] = {
            "variant": name,
            "vocab_size_target": args.vocab_size,
            "vocab_size_actual": actual,
            "model": args.model,
            "langs": VARIANT_LANGS.get(name, VARIANT_LANGS["V3"]),
            "alpha": (float(suffix[1:]) if suffix else None),
            "train_lines": len(text),
            "max_lines_per_lang": args.max_lines_per_lang,
            "seed": args.seed,
            "tokenizer_sha256_16": digest,
        }
        print(f"   → {d}  (actual vocab {actual:,}, sha256 {digest})")

    (args.out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{len(index)} variants total → {args.out_dir}")


if __name__ == "__main__":
    main()
