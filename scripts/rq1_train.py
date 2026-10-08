#!/usr/bin/env python3
"""RQ1 as a script, for a cluster: fine-tune NLLB with one helper language, translate the test set, save scores.

Same steps and settings as notebooks/rq1_helpers.ipynb, see src/helper_training.py for the design.

    python scripts/rq1_train.py --run uk --seed 43    # runs: none, uk, ru_matched, ru
    python scripts/rq1_train.py --run uk --smoke      # 50 steps, 50 test sentences per direction
    python scripts/rq1_train.py --run uk --max-steps 30000 --patience 0 --tag long --also-last
                                                      # no early stopping; score best-on-dev AND final model
    python scripts/rq1_train.py --run tr --from-scratch    # same data and evaluation, but a small NLLB-shaped
                                                      # model with random weights (no pretraining) -> <system>_scratch

Outputs: <ckpt-root>/<system>/best, results/translations/<system>/, results/metrics/<system>/
with <system> = nllb_helper_<run>_s<seed> (plus _smoke).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.helper_training import RUNS, system_name  # noqa: E402

MODEL_NAME = "facebook/nllb-200-distilled-600M"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--alpha", type=float, default=0.2, help="temperature sampling; keep the same for all runs")
    ap.add_argument("--max-steps", type=int, help="optimizer updates; keep the same for all runs "
                    "(default 10,000; 40,000 with --from-scratch)")
    ap.add_argument("--eval-steps", type=int, help="default 500; 1,000 with --from-scratch")
    ap.add_argument("--patience", type=int,
                    help="evaluations without dev-loss improvement before stopping; 0 = never stop early "
                    "(default 4; 5 with --from-scratch)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="random weights instead of pretrained NLLB: a 6-layer, 512-wide model with NLLB's "
                    "architecture and tokenizer, trained with from-scratch settings (lr 5e-4, dropout 0.3, "
                    "label smoothing 0.1, batch 64); folder gets the tag 'scratch'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint on disk")
    ap.add_argument("--ckpt-root", type=Path, default=ROOT / "artifacts" / "checkpoints",
                    help="where checkpoints go (~15 GB per run); on a cluster use the node's scratch disk")
    ap.add_argument("--smoke", action="store_true", help="50 steps, 50 test sentences per direction")
    ap.add_argument("--tag", default="", help="added to the folder name, e.g. long -> nllb_helper_uk_s42_long")
    ap.add_argument("--also-last", action="store_true",
                    help="also translate with the final checkpoint, into <system>_last")
    args = ap.parse_args()
    for name, pretrained, scratch in (("max_steps", 10_000, 40_000), ("eval_steps", 500, 1_000), ("patience", 4, 5)):
        if getattr(args, name) is None:
            setattr(args, name, scratch if args.from_scratch else pretrained)
    if args.from_scratch and not args.tag:
        args.tag = "scratch"

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments

    from src.data import TedData
    from src.helper_training import (NLLB_CODES, TARGET_DIRECTIONS, build_mixture, make_trainer,
                                     mixture_report, translate_and_save)

    if args.smoke:
        args.max_steps, args.eval_steps = 50, 25
    system = system_name(args.run, args.seed) + (f"_{args.tag}" if args.tag else "") + ("_smoke" if args.smoke else "")
    ckpt_dir = args.ckpt_root / system
    out_dir = ROOT / "results" / "translations" / system
    metrics_dir = ROOT / "results" / "metrics" / system
    for d in (ckpt_dir, out_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible: run this inside a GPU job")
    gpu = torch.cuda.get_device_name(0)
    bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    # same as baselines_nllb.ipynb: effective batch 32
    batch, accum, grad_ckpt = (16, 2, False) if bf16 else (4, 8, True)
    lr, max_len, label_smoothing = 5e-5, 128, 0.0
    if args.from_scratch:  # small-data settings for a model without pretraining (Sennrich & Zhang 2019)
        batch, accum, grad_ckpt, lr, label_smoothing = 64, 1, False, 5e-4, 0.1
    print(f"GPU: {gpu} | {'bf16' if bf16 else 'fp16'} | run: {system}", flush=True)

    ted = TedData(download=False)
    groups = build_mixture(ted, args.run, seed=args.seed)
    dev_rows = ted.examples("dev", TARGET_DIRECTIONS, NLLB_CODES)
    test_rows = ted.examples("test", TARGET_DIRECTIONS, NLLB_CODES)
    if args.smoke:
        test_rows = [r for d in TARGET_DIRECTIONS for r in [x for x in test_rows if x["direction"] == d][:50]]
    print(mixture_report(groups, args.alpha), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if args.from_scratch:
        # NLLB's architecture and vocabulary, scaled down, randomly initialised: the only difference from the
        # pretrained runs is that the weights know nothing about any language
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.update({"d_model": 512, "encoder_layers": 6, "decoder_layers": 6,
                       "encoder_attention_heads": 8, "decoder_attention_heads": 8,
                       "encoder_ffn_dim": 2048, "decoder_ffn_dim": 2048,
                       "dropout": 0.3, "attention_dropout": 0.1, "activation_dropout": 0.1})
        torch.manual_seed(args.seed)
        model = AutoModelForSeq2SeqLM.from_config(config)
        print(f"from scratch: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters", flush=True)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    if grad_ckpt:
        model.gradient_checkpointing_enable()

    targs = Seq2SeqTrainingArguments(
        output_dir=ckpt_dir, max_steps=args.max_steps,
        learning_rate=lr, per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum, warmup_steps=int(0.05 * args.max_steps), weight_decay=0.01,
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        bf16=bf16, fp16=not bf16, seed=args.seed, logging_steps=50, report_to="none",
        dataloader_num_workers=2, disable_tqdm=True,
    )
    trainer = make_trainer(model, tok, groups, dev_rows, targs, alpha=args.alpha, max_len=max_len,
                           patience=args.patience, label_smoothing=label_smoothing)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(f"{ckpt_dir}/best")
    tok.save_pretrained(f"{ckpt_dir}/best")
    print("best checkpoint:", trainer.state.best_model_checkpoint, "| best dev loss:", trainer.state.best_metric,
          flush=True)

    run_info = {
        "run": args.run, "helper": RUNS[args.run][0], "helper_cap": RUNS[args.run][1],
        "rows_per_group": {g: len(r) for g, r in groups.items()},
        "model": MODEL_NAME, "transformers_version": transformers.__version__, "gpu": gpu,
        "precision": "bf16" if bf16 else "fp16", "seed": args.seed, "lr": lr, "alpha": args.alpha,
        "max_steps": args.max_steps, "steps_done": trainer.state.global_step,
        "batch": batch, "grad_accum": accum, "effective_batch": batch * accum,
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_dev_loss": trainer.state.best_metric,
        "patience": args.patience, "scored_checkpoint": "best",
        "init": "scratch (6x512, random weights)" if args.from_scratch else "pretrained",
        "label_smoothing": label_smoothing,
    }

    def score(model, out, metrics, info):
        scores = translate_and_save(model, tok, test_rows, out, max_len=max_len)
        metrics.mkdir(parents=True, exist_ok=True)
        (metrics / "scores.json").write_text(json.dumps(scores, indent=2))
        (metrics / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
        (metrics / "run_info.json").write_text(json.dumps(info, indent=2))
        print(ted.check_references(out), flush=True)  # every .ref must say OK: same test set as every other system

    # load_best_model_at_end: trainer.model is already the best checkpoint
    score(trainer.model, out_dir, metrics_dir, run_info)

    if args.also_last:
        # save_total_limit keeps the best and the most recent checkpoint; the most recent is the final model
        last = max(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        print("final checkpoint:", last, flush=True)
        final = AutoModelForSeq2SeqLM.from_pretrained(last).cuda()
        score(final, ROOT / "results" / "translations" / f"{system}_last", ROOT / "results" / "metrics" / f"{system}_last",
              {**run_info, "scored_checkpoint": f"last ({last.name})"})


if __name__ == "__main__":
    main()
