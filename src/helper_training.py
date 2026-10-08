"""Fine-tune NLLB with a helper language mixed in, rebalanced with temperature sampling (RQ1).

RQ1 asks whether Belarusian is helped more by Ukrainian (more similar) or by Russian (more
data). Russian has about twice Ukrainian's pairs, so `+ru` against `+uk` alone mixes up the two
factors. The runs in `RUNS` separate them:

    none        no helper: the reference every helper run is compared with
    uk          Ukrainian, all 108,495 pairs
    ru_matched  Russian, subsampled to Ukrainian's pair count  -> compare with uk: similarity
    ru          Russian, all 208,458 pairs                      -> compare with ru_matched: volume
    tr          Turkish, all 182,470 pairs: Azerbaijani's helper (RQ2: training-data side, vs pivoting)
    none_az1k   no helper, Azerbaijani cut to 1,000 pairs      } does Turkish help when Azerbaijani is
    tr_az1k     Turkish, Azerbaijani cut to 1,000 pairs        } truly scarce? (same az subset per seed)

Every run trains on the baseline's four directions (az-en, en-az, be-en, en-be: the same data as
`nllb_finetuned_no_helpers` in `baselines_nllb.ipynb`) plus the helper's two directions, so the
helper is the only difference from `none`. Every run gets the same number of updates
(`max_steps`), so a bigger helper does not also mean more training. Each setting runs with every
seed in `SEEDS`; the seed also picks `ru_matched`'s Russian subset.

Batches come from `TemperatureBatchSampler` (src/preprocessing/temperature_sampler.py): each batch
is one language pair, picked with p ∝ n^alpha. The pure-Python parts (`build_mixture`,
`MixtureBatchSampler`) need no GPU and are tested in tests/test_helper_training.py; torch and
transformers are imported only by the functions that train and translate.
"""

from __future__ import annotations

import random
from pathlib import Path

from src.data import TedData, align  # noqa: F401  (align also puts src/preprocessing on sys.path)
from temperature_sampler import TemperatureBatchSampler, language_weights

NLLB_CODES = {"en": "eng_Latn", "az": "azj_Latn", "be": "bel_Cyrl",
              "tr": "tur_Latn", "uk": "ukr_Cyrl", "ru": "rus_Cyrl"}

# The baseline's directions; dev and test use only these, so every run is scored like the baseline.
TARGET_DIRECTIONS = ["az-en", "en-az", "be-en", "en-be"]

# run name -> (helper language, cap on its training pairs: None = all, a language = that language's count)
RUNS = {
    "none": (None, None),
    "uk": ("uk", None),
    "ru_matched": ("ru", "uk"),
    "ru": ("ru", None),
    "tr": ("tr", None),
    "none_az1k": (None, None),
    "tr_az1k": ("tr", None),
}
# run name -> {target language: pairs kept}; the subset depends only on the seed, so runs that share a
# seed share it, and the helper stays the only difference between them
TARGET_CAPS = {"none_az1k": {"az": 1000}, "tr_az1k": {"az": 1000}}
SEEDS = [42, 43, 44]


def system_name(run: str, seed: int) -> str:
    """Folder name under results/translations/ and artifacts/checkpoints/, e.g. `nllb_helper_uk_s42`."""
    return f"nllb_helper_{run}_s{seed}"


def _directions(a: str, b: str) -> list[str]:
    return [f"{a}-{b}", f"{b}-{a}"]


def build_mixture(ted: TedData, run: str, seed: int = 42) -> dict[str, list[dict]]:
    """Training rows grouped by sampling group: `{"az": [...], "be": [...], helper: [...]}`.

    Each group holds both directions of its pair with English, as NLLB-style rows (see
    `TedData.examples`). A capped helper or target keeps a seeded random subset of its pairs, the
    same subset for both directions.
    """
    helper, cap = RUNS[run]
    groups = {}
    for low in ("az", "be"):
        n = TARGET_CAPS.get(run, {}).get(low)
        if n is None:
            groups[low] = ted.examples("train", _directions(low, "en"), NLLB_CODES)
            continue
        pairs = ted.pairs("train", low, "en")
        keep = set(random.Random(f"{low}-{seed}").sample(range(len(pairs)), n))
        groups[low] = [row for i, (_, x, e) in enumerate(pairs) if i in keep for row in (
            {"src": x, "tgt": e, "src_lang": NLLB_CODES[low], "tgt_lang": NLLB_CODES["en"], "direction": f"{low}-en"},
            {"src": e, "tgt": x, "src_lang": NLLB_CODES["en"], "tgt_lang": NLLB_CODES[low], "direction": f"en-{low}"},
        )]
    if helper is None:
        return groups

    pairs = ted.pairs("train", helper, "en")
    if cap is not None:
        n = len(ted.pairs("train", cap, "en"))
        if n > len(pairs):
            raise ValueError(f"cannot cap {helper} ({len(pairs)} pairs) at {cap}'s {n}")
        keep = set(random.Random(seed).sample(range(len(pairs)), n))
        pairs = [p for i, p in enumerate(pairs) if i in keep]
    groups[helper] = [row for _, h, e in pairs for row in (
        {"src": h, "tgt": e, "src_lang": NLLB_CODES[helper], "tgt_lang": NLLB_CODES["en"], "direction": f"{helper}-en"},
        {"src": e, "tgt": h, "src_lang": NLLB_CODES["en"], "tgt_lang": NLLB_CODES[helper], "direction": f"en-{helper}"},
    )]
    return groups


def mixture_report(groups: dict[str, list[dict]], alpha: float) -> str:
    """One line per group: rows and the share of batches it gets at `alpha`. Print it before training."""
    sizes = {k: len(v) for k, v in groups.items()}
    w = language_weights(sizes, alpha)
    return "\n".join(f"{k:4s} {n:>8,} rows   {w[k]:6.1%} of batches" for k, n in sizes.items())


class MixtureBatchSampler:
    """`TemperatureBatchSampler` over the concatenation of `groups`, for a DataLoader's `batch_sampler`.

    Yields lists of global indices into `[*groups[g1], *groups[g2], ...]` (dict order), one group per
    batch. Each pass (one Trainer epoch) uses a new seed, so the order of groups differs between passes.
    """

    def __init__(self, groups: dict[str, list], alpha: float, batch_size: int, steps: int, seed: int = 42):
        self.offsets, start = {}, 0
        for g, rows in groups.items():
            self.offsets[g] = start
            start += len(rows)
        self.sampler = TemperatureBatchSampler({g: len(r) for g, r in groups.items()}, alpha=alpha,
                                               batch_size=batch_size, seed=seed, steps_per_epoch=steps)
        self.seed = seed
        self.batch_size = batch_size  # read by accelerate when it wraps the DataLoader
        self._passes = 0

    def __iter__(self):
        self.sampler.seed = self.seed + self._passes
        self._passes += 1
        for g, idx in self.sampler:
            yield [self.offsets[g] + i for i in idx]

    def __len__(self) -> int:
        return len(self.sampler)


# ------------------------------------------------------------------ GPU side (torch / transformers)
def encode_rows(tok, rows: list[dict], max_len: int = 128):
    """Tokenize NLLB-style rows into a `datasets.Dataset`, in the same order as `rows`.

    Tokenizes one direction at a time in batches (the language tags are tokenizer state, so they
    cannot change inside a batch), which is much faster than `Dataset.map` row by row.
    """
    from datasets import Dataset

    out = {"input_ids": [None] * len(rows), "attention_mask": [None] * len(rows), "labels": [None] * len(rows)}
    by_dir: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(rows):
        by_dir.setdefault((r["src_lang"], r["tgt_lang"]), []).append(i)
    for (src_lang, tgt_lang), idx in by_dir.items():
        tok.src_lang, tok.tgt_lang = src_lang, tgt_lang
        for s in range(0, len(idx), 2000):
            chunk = idx[s:s + 2000]
            enc = tok([rows[i]["src"] for i in chunk], text_target=[rows[i]["tgt"] for i in chunk],
                      max_length=max_len, truncation=True)
            for j, i in enumerate(chunk):
                for k in out:
                    out[k][i] = enc[k][j]
    return Dataset.from_dict(out)


def make_trainer(model, tok, groups: dict[str, list[dict]], dev_rows: list[dict], args, alpha: float,
                 max_len: int = 128, patience: int = 4, label_smoothing: float = 0.0):
    """A `Seq2SeqTrainer` whose training batches come from `MixtureBatchSampler`.

    `args` is a `Seq2SeqTrainingArguments` with `max_steps` set; the sampler is sized to exactly
    `max_steps` optimizer updates. `patience=0` turns early stopping off (train all `max_steps`).

    `label_smoothing` applies to the training loss only; the dev loss used for early stopping stays plain
    cross-entropy, as in the runs without it. It is computed here rather than through
    `label_smoothing_factor`: transformers 5.x's label smoother pops the labels and then crashes M2M100
    ("cannot specify both decoder_input_ids and decoder_inputs_embeds").
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer

    if args.max_steps <= 0:
        raise ValueError("set max_steps: every run gets the same number of updates")
    train = encode_rows(tok, [r for rows in groups.values() for r in rows], max_len)
    dev = encode_rows(tok, dev_rows, max_len)
    batch_sampler = MixtureBatchSampler(groups, alpha, args.per_device_train_batch_size,
                                        steps=args.max_steps * args.gradient_accumulation_steps, seed=args.seed)

    class MixtureTrainer(Seq2SeqTrainer):
        def get_train_dataloader(self):
            dl = DataLoader(self.train_dataset, batch_sampler=batch_sampler, collate_fn=self.data_collator,
                            num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory)
            return self.accelerator.prepare(dl)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if label_smoothing <= 0 or not model.training:
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            outputs = model(**inputs)  # labels stay in: the model builds decoder_input_ids as usual
            labels = inputs["labels"]
            loss = F.cross_entropy(outputs.logits.float().view(-1, outputs.logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, label_smoothing=label_smoothing, reduction="sum")
            loss = loss / (num_items_in_batch if num_items_in_batch is not None else (labels != -100).sum())
            return (loss, outputs) if return_outputs else loss

    return MixtureTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                          data_collator=DataCollatorForSeq2Seq(tok, model=model),
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)] if patience > 0 else [])


def translate_and_save(model, tok, rows: list[dict], out_dir, batch: int = 32, beams: int = 5,
                       max_len: int = 128) -> dict:
    """Translate the test rows per direction, write `<dir>.src/.ref/.hyp` into `out_dir`, return BLEU/chrF.

    Same decoding as `translate_and_score` in `baselines_nllb.ipynb`, so the scores are comparable.
    """
    import sacrebleu
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    model.eval()
    scores = {}
    for d in dict.fromkeys(r["direction"] for r in rows):
        sub = [r for r in rows if r["direction"] == d]
        tok.src_lang = sub[0]["src_lang"]
        bos = tok.convert_tokens_to_ids(sub[0]["tgt_lang"])
        hyps = []
        for i in range(0, len(sub), batch):
            enc = tok([r["src"] for r in sub[i:i + batch]], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(model.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams,
                                     max_new_tokens=max_len, max_length=None)
            hyps += tok.batch_decode(gen, skip_special_tokens=True)
        refs = [r["tgt"] for r in sub]
        for ext, lines in (("src", [r["src"] for r in sub]), ("ref", refs), ("hyp", hyps)):
            (out_dir / f"{d}.{ext}").write_text("\n".join(lines) + "\n", encoding="utf-8")
        scores[d] = {"BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
                     "chrF": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)}
        print(d, scores[d])
    return scores
