#!/usr/bin/env python3
"""Is an RQ1 run finished, healthy, and are its translations sane? Prints one block per run, then a score table.

    python scripts/check_rq1.py                 # every nllb_helper_* run on disk (smoke runs too)
    python scripts/check_rq1.py nllb_helper_uk_s42  # just these

Lines start with OK, WARN (look at it) or FAIL (do not use these numbers). The last table averages
each setting over its seeds: mean ± spread between seeds, and the change against `none` (no helper).
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METRICS = ROOT / "results" / "metrics"
TRANS = ROOT / "results" / "translations"
DIRECTIONS = ["az-en", "en-az", "be-en", "en-be"]
BASELINES = ["nllb_basemodel", "nllb_finetuned_no_helpers"]

# Letters Russian/Ukrainian use and Belarusian never does: Belarusian output containing them has
# drifted into the helper language, the typical failure of helper training.
NOT_BELARUSIAN = re.compile(r"[ищъїєґ]", re.IGNORECASE)
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")


def lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8").splitlines() if p.exists() else []


def check_training(system: str, say) -> None:
    info_f, hist_f = METRICS / system / "run_info.json", METRICS / system / "log_history.json"
    if not info_f.exists():
        say("FAIL", "no run_info.json: the job did not finish (see logs/rq1_*.out)")
        return
    info = json.loads(info_f.read_text())
    done, planned = info["steps_done"], info["max_steps"]
    say("OK" if done == planned else "WARN",
        f"steps {done:,} / {planned:,}" + ("" if done == planned else " (early stopping: fine if dev loss had flattened)"))
    say("OK", f"alpha {info['alpha']}, seed {info['seed']}, lr {info['lr']}, effective batch {info['effective_batch']}, "
              f"rows {info['rows_per_group']}")

    hist = json.loads(hist_f.read_text()) if hist_f.exists() else []
    train = [h["loss"] for h in hist if "loss" in h]
    dev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    if any(not math.isfinite(x) for x in train + [d for _, d in dev]):
        say("FAIL", "loss became NaN/inf: training diverged")
    if len(train) >= 4:
        a, b = sum(train[:3]) / 3, sum(train[-3:]) / 3
        say("OK" if b < a else "FAIL", f"train loss {a:.3f} -> {b:.3f}")
    if dev:
        best_step, best = min(dev, key=lambda x: x[1])
        curve = "  ".join(f"{s}:{d:.3f}" for s, d in dev)
        say("OK" if best < dev[0][1] or len(dev) == 1 else "WARN", f"dev loss (step:loss) {curve}")
        if best_step == dev[-1][0] and info["steps_done"] == info["max_steps"]:
            say("WARN", "dev loss was still falling at the last step: more steps could help (keep runs equal)")


def check_translations(system: str, say, ted) -> None:
    folder = TRANS / system
    if not folder.exists():
        say("FAIL", "no translations folder")
        return
    if ted is not None:
        bad = {k: v for k, v in ted.check_references(folder).items() if not v.startswith("OK")}
        say("FAIL" if bad else "OK", f"references differ from the official test set: {bad}" if bad
            else "references = the official test set (same as every other system)")
    for d in DIRECTIONS:
        src, ref, hyp = (lines(folder / f"{d}.{e}") for e in ("src", "ref", "hyp"))
        if not hyp:
            say("FAIL", f"{d}: no hypotheses")
            continue
        n = len(hyp)
        problems = []
        if len(ref) != n:
            problems.append(f"{n} hyps vs {len(ref)} refs")
        empty = sum(not h.strip() for h in hyp) / n
        copied = sum(h.strip() == s.strip() for h, s in zip(hyp, src)) / n
        tgt = d.split("-")[1]
        script = CYRILLIC if tgt == "be" else LATIN
        wrong_script = sum(bool(h.strip()) and not script.search(h) for h in hyp) / n
        for share, what, limit in ((empty, "empty", 0.01), (copied, "copied from the source", 0.02),
                                   (wrong_script, "in the wrong script", 0.02)):
            if share > limit:
                problems.append(f"{share:.1%} {what}")
        detail = f"{n} lines, {empty:.1%} empty, {copied:.1%} copied, {wrong_script:.1%} wrong script"
        if tgt == "be":
            leak = sum(bool(NOT_BELARUSIAN.search(h)) for h in hyp) / n
            ref_leak = sum(bool(NOT_BELARUSIAN.search(r)) for r in ref) / max(1, len(ref))
            detail += f", {leak:.1%} with ru/uk-only letters (references: {ref_leak:.1%})"
            if leak > ref_leak + 0.05:
                problems.append(f"{leak:.1%} of Belarusian output has ru/uk-only letters: drifting into the helper")
        say("FAIL" if any("hyps vs" in p or "empty" in p for p in problems) else "WARN" if problems else "OK",
            f"{d}: " + ("; ".join(problems) if problems else detail))


def score_table(systems: list[str]) -> None:
    rows = []
    for s in BASELINES + systems:
        f = METRICS / s / "scores.json"
        if f.exists():
            sc = json.loads(f.read_text())
            rows.append((s, [sc.get(d, {}) for d in DIRECTIONS]))
    if not rows:
        return
    print("\nBLEU / chrF per direction" + ("" if any(r[0] in BASELINES for r in rows)
                                            else "   (baseline scores not on disk: copy them into results/metrics/)"))
    print(f"{'system':32s}" + "".join(f"{d:>15s}" for d in DIRECTIONS))
    for s, sc in rows:
        print(f"{s:32s}" + "".join(f"{x.get('BLEU', float('nan')):>7.1f} /{x.get('chrF', float('nan')):>5.1f}" for x in sc))


def seed_table(systems: list[str]) -> None:
    """Mean ± standard deviation over seeds per setting, BLEU then chrF, plus the change against `none`."""
    by_run: dict[str, list[dict]] = {}
    for s in systems:
        m = re.fullmatch(r"nllb_helper_(.+?)_s(\d+)(_scratch)?", s)
        f = METRICS / s / "scores.json"
        if m and f.exists():
            sc = json.loads(f.read_text())
            cf = METRICS / s / "comet.json"  # from scripts/comet_rq1.py; shown x100
            if cf.exists():
                for d, c in json.loads(cf.read_text()).items():
                    sc[d]["COMET"] = 100 * c["score"]
            by_run.setdefault(m.group(1) + (m.group(3) or ""), []).append(sc)
    if not by_run:
        return
    known = ("none", "uk", "ru_matched", "ru", "tr", "none_az1k", "tr_az1k",
             "none_scratch", "uk_scratch", "ru_matched_scratch", "ru_scratch", "tr_scratch")
    order = [r for r in known if r in by_run] + sorted(set(by_run) - set(known))
    metrics = ["BLEU", "chrF"] + (["COMET"] if all("COMET" in sc[d] for v in by_run.values() for sc in v for d in DIRECTIONS) else [])
    mean = {(r, d, m): statistics.mean(sc[d][m] for sc in by_run[r]) for r in order for d in DIRECTIONS for m in metrics}
    for metric in metrics:
        print(f"\n{metric}, averaged over seeds (± = spread between seeds; [x] = change vs none, or vs none_az1k for _az1k)")
        print(f"{'setting':12s}{'seeds':>6s}" + "".join(f"{d:>22s}" for d in DIRECTIONS))
        for r in order:
            cells = []
            for d in DIRECTIONS:
                vals = [sc[d][metric] for sc in by_run[r]]
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                ref = next((f"none{k}" for k in ("_az1k", "_scratch") if r.endswith(k)), "none")  # like with like
                delta = f" [{mean[(r, d, metric)] - mean[(ref, d, metric)]:+.1f}]" if ref in by_run and r != ref else ""
                cells.append(f"{statistics.mean(vals):.1f} ±{sd:.1f}{delta}")
            print(f"{r:12s}{len(by_run[r]):>6d}" + "".join(f"{c:>22s}" for c in cells))


def main() -> None:
    systems = sys.argv[1:] or sorted(p.name for p in METRICS.glob("nllb_helper_*") if p.is_dir())
    if not systems:
        raise SystemExit(f"no nllb_helper_* runs under {METRICS}")
    try:
        from src.data import TedData
        ted = TedData(download=False)
    except Exception as e:  # noqa: BLE001  (no corpus here: skip the reference check, say so)
        print(f"(reference check skipped: {e})")
        ted = None
    for s in systems:
        print(f"\n=== {s}")
        say = lambda level, msg: print(f"  {level:4s} {msg}")
        check_training(s, say)
        check_translations(s, say, ted)
    real = [s for s in systems if not s.endswith("_smoke")]
    score_table(real)
    seed_table(real)


if __name__ == "__main__":
    main()
