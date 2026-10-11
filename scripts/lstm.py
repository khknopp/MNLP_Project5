import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import get_language_data_aligned
from src.evaluation import evaluate_direction
from src.model import AttentionSeq2Seq, Seq2SeqModel, Vocabulary, save_translations, translate_corpus

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--directions", nargs="+", required=True)
parser.add_argument("--pivot")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--arch", choices=["attn", "basic"], default="attn")
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--max-length", type=int, default=None)
parser.add_argument("--max-pairs", type=int, default=0, help="cap the training pairs per model (0 = all); useful for the huge pivot corpora")
parser.add_argument("--vocab-size", type=int, default=4000)
args = parser.parse_args()
args.epochs = args.epochs or (50 if args.arch == "attn" else 10)
args.max_length = args.max_length or (100 if args.arch == "attn" else 20)

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(args.seed)
torch.manual_seed(args.seed)
run = f"{args.name}{'' if args.arch == 'basic' else '_attn'}_s{args.seed}"
out_dir = ROOT / "results" / "translations" / run
metrics_dir = ROOT / "results" / "metrics" / run
metrics_dir.mkdir(parents=True, exist_ok=True)


_models = {}


def train(src, tgt):
    # a pivot run needs the same model for several directions (e.g. tr-en for both az-en and be-en): train it once
    if (src, tgt) not in _models:
        _models[(src, tgt)] = _train(src, tgt)
    return _models[(src, tgt)]


def _train(src, tgt):
    train_pairs = get_language_data_aligned("train", src, tgt)
    if args.max_pairs and len(train_pairs) > args.max_pairs:
        train_pairs = random.Random(args.seed).sample(train_pairs, args.max_pairs)
    print(f"training {src}-{tgt} on {len(train_pairs)} pairs", flush=True)
    if args.arch == "attn":
        model = AttentionSeq2Seq.from_pairs(train_pairs, vocab_size=args.vocab_size)
        model.fit(pairs=train_pairs, dev_pairs=get_language_data_aligned("dev", src, tgt), epochs=args.epochs, device=device)
        return model
    model = Seq2SeqModel(Vocabulary([s for s, _ in train_pairs]), Vocabulary([t for _, t in train_pairs]))
    model.fit(pairs=train_pairs, epochs=args.epochs, device=device)
    return model


def translate(model, sources):
    return translate_corpus(model, sources, max_length=args.max_length, device=device)


scores = {}
for direction in args.directions:
    src, tgt = direction.split("-")
    test = get_language_data_aligned("test", src, tgt)
    sources = [s for s, _ in test]
    if args.pivot:
        hyps = translate(train(args.pivot, tgt), translate(train(src, args.pivot), sources))
    else:
        hyps = translate(train(src, tgt), sources)
    save_translations(out_dir, direction, sources, [t for _, t in test], hyps)
    scores[direction] = {m: evaluate_direction(out_dir, direction, m, gpus=int(device == "cuda")) for m in ("bleu", "chrf", "comet")}
    json.dump({"args": vars(args), "scores": scores}, open(metrics_dir / "scores.json", "w"), indent=2)

print(json.dumps({d: {m: r["score"] for m, r in v.items()} for d, v in scores.items()}, indent=2))
