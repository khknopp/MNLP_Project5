#!/usr/bin/env python3
"""COMET for the RQ1 runs, from the saved .src/.hyp/.ref files: no retraining, no re-translating.

    python scripts/comet_rq1.py                  # every nllb_helper_<run>_s<seed> folder under results/
    python scripts/comet_rq1.py --gpus 0         # on CPU (slow)

Scores through src/evaluation.py, so the numbers are computed exactly like the rest of the team's.
Writes results/metrics/<system>/comet.json: {direction: {"score": ..., "sentence_scores": [...]}}.
Sentence scores are kept because scripts/significance_rq1.py bootstraps over them. Systems that already
have a comet.json are skipped, so an interrupted job can simply be resubmitted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import COMET_MODEL, evaluate_all_directions, load_metric  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    systems = sorted(d for d in (args.results / "translations").iterdir()
                     if re.fullmatch(r"nllb_helper_.+_s\d+(_scratch)?", d.name))
    todo = [d for d in systems if not (args.results / "metrics" / d.name / "comet.json").exists()]
    print(f"{len(systems)} systems, {len(todo)} still to score with {COMET_MODEL}", flush=True)
    if not todo:
        return

    metric = load_metric("comet")
    for folder in todo:
        res = evaluate_all_directions(folder, "comet", metric=metric, batch_size=args.batch_size, gpus=args.gpus)
        out = {d: {"score": r["score"], "sentence_scores": r["sentence_scores"]} for d, r in res.items()}
        target = args.results / "metrics" / folder.name
        target.mkdir(parents=True, exist_ok=True)
        (target / "comet.json").write_text(json.dumps(out))
        print(folder.name, {d: round(r["score"], 4) for d, r in res.items()}, flush=True)


if __name__ == "__main__":
    main()
