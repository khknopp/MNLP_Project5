import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import get_language_data_aligned
from src.evaluation import evaluate_direction
from src.model import Seq2SeqModel, Vocabulary, save_translations, translate_corpus

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--directions", nargs="+", required=True)
parser.add_argument("--pivot")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--max-length", type=int, default=20)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = ROOT / "results" / "translations" / args.name
metrics_dir = ROOT / "results" / "metrics" / args.name
metrics_dir.mkdir(parents=True, exist_ok=True)


def train(src, tgt):
    train_pairs = get_language_data_aligned("train", src, tgt)
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
    scores[direction] = {m: evaluate_direction(out_dir, direction, m, gpus=int(device == "cuda"))["score"] for m in ("bleu", "chrf", "comet")}

json.dump({"args": vars(args), "scores": scores}, open(metrics_dir / "scores.json", "w"), indent=2)
print(json.dumps(scores, indent=2))
