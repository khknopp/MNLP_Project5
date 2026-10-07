import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import TedData
from src.evaluation import DIRECTIONS, evaluate_direction
from src.model import NLLB_CODES, save_translations
from temperature_sampler import language_weights

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--helpers", nargs="*", default=[])
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--direct", nargs="*", default=[], help="extra direct directions to train on, e.g. tr-az az-tr")
parser.add_argument("--match-budget", action="store_true", help="train on as many examples as the target-only run (equal compute)")
parser.add_argument("--select-target", action="store_true", help="early-stop / pick the best epoch on the az/be dev sets only")
parser.add_argument("--stage2-epochs", type=int, default=0, help="afterwards, polish on target-language data only")
parser.add_argument("--base", action="store_true")
parser.add_argument("--pivot")
parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

MAX_LEN, BEAMS, BATCH, ACCUM = 128, 5, 16, 2
BF16 = torch.cuda.get_device_capability(0)[0] >= 8
RUN = f"{args.name}_s{args.seed}"
CKPT_DIR = ROOT / "artifacts" / "checkpoints" / RUN
OUT_DIR = ROOT / "results" / "translations" / RUN
METRICS_DIR = ROOT / "results" / "metrics" / RUN
METRICS_DIR.mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

set_seed(args.seed)

tok = AutoTokenizer.from_pretrained(args.model)
ted = TedData()


def translate(model, texts, src, tgt, batch=32):
    tok.src_lang = NLLB_CODES[src]
    bos = tok.convert_tokens_to_ids(NLLB_CODES[tgt])
    hyps = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16 if BF16 else torch.float16):
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=BEAMS, max_new_tokens=MAX_LEN, max_length=None)
        hyps += tok.batch_decode(gen, skip_special_tokens=True)
    return hyps


if not args.base and not args.pivot:
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

    train_dirs = list(DIRECTIONS) + [d for h in args.helpers for d in (f"{h}-en", f"en-{h}")] + args.direct
    rows = {d: ted.examples("train", [d], NLLB_CODES) for d in train_dirs}
    rng = random.Random(args.seed)
    total = sum(len(rows[d]) for d in DIRECTIONS) if args.match_budget else sum(len(r) for r in rows.values())
    weights = language_weights({d: len(r) for d, r in rows.items()}, args.alpha)
    train_rows = [x for d, r in rows.items() for x in rng.choices(r, k=round(total * weights[d]))]
    rng.shuffle(train_rows)
    dev_rows = ted.examples("dev", list(DIRECTIONS) if args.select_target else train_dirs, NLLB_CODES)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def encode(ex):
        tok.src_lang, tok.tgt_lang = ex["src_lang"], ex["tgt_lang"]
        return tok(ex["src"], text_target=ex["tgt"], max_length=MAX_LEN, truncation=True)

    def run_stage(train_rows, dev_rows, epochs, lr, out_dir):
        train_ds = Dataset.from_list(train_rows).map(encode, remove_columns=list(train_rows[0]))
        dev_ds = Dataset.from_list(dev_rows).map(encode, remove_columns=list(dev_rows[0]))
        steps_per_epoch = -(-len(train_ds) // (BATCH * ACCUM))
        trainer = Seq2SeqTrainer(
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir=str(out_dir), learning_rate=lr, per_device_train_batch_size=BATCH,
                per_device_eval_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
                num_train_epochs=epochs, warmup_steps=int(0.05 * steps_per_epoch * epochs), weight_decay=0.01,
                eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                load_best_model_at_end=True, metric_for_best_model="eval_loss",
                bf16=BF16, fp16=not BF16, seed=args.seed, logging_steps=50, report_to="none", group_by_length=True,
            ),
            train_dataset=train_ds, eval_dataset=dev_ds,
            data_collator=DataCollatorForSeq2Seq(tok, model=model),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        return trainer

    trainer = run_stage(train_rows, dev_rows, args.epochs, args.lr, CKPT_DIR)
    history = trainer.state.log_history
    if args.stage2_epochs:
        target_rows = [x for d in DIRECTIONS for x in rows[d]]
        rng.shuffle(target_rows)
        trainer = run_stage(target_rows, ted.examples("dev", list(DIRECTIONS), NLLB_CODES), args.stage2_epochs, args.lr / 5, CKPT_DIR / "stage2")
        history = history + trainer.state.log_history
    trainer.save_model(str(CKPT_DIR / "best"))
    tok.save_pretrained(str(CKPT_DIR / "best"))
    json.dump(history, open(METRICS_DIR / "log_history.json", "w"), indent=2)
    model = trainer.model
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).cuda()
model.eval()

for direction in DIRECTIONS:
    src, tgt = direction.split("-")
    pairs = [(s, t) for _, s, t in ted.pairs("test", src, tgt)]
    sources = [s for s, _ in pairs]
    if args.pivot:
        hyps = translate(model, translate(model, sources, src, args.pivot), args.pivot, tgt)
    else:
        hyps = translate(model, sources, src, tgt)
    save_translations(OUT_DIR, direction, sources, [t for _, t in pairs], hyps)

scores = {}
for direction in DIRECTIONS:
    scores[direction] = {m: evaluate_direction(OUT_DIR, direction, m, gpus=1) for m in ("bleu", "chrf", "comet")}
    json.dump({"args": vars(args), "scores": scores}, open(METRICS_DIR / "scores.json", "w"), indent=2)
print(json.dumps({d: {m: r["score"] for m, r in v.items()} for d, v in scores.items()}, indent=2))
