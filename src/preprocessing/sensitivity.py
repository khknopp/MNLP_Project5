#!/usr/bin/env python3
"""
WP0 · Sensitivity — does any conclusion depend on an analysis setting?

Every number in this project rests on settings someone chose: how many helper sentences define
the shared-piece set, what counts as a "productive affix", which split is evaluated, how the
corpus is sampled for the allocation metric. If a conclusion moves when a setting moves, then it
is a statement about the setting, not about the data — and **nothing in the pipeline would ever
say so**. This script says so.

It deliberately does **not** reimplement the analysis. It re-runs the real `cost_analysis.py`
under different flags and reads the CSVs back, because a second implementation would drift from
the first and then both would be wrong in different ways. The cross-check in `directionality.py`
exists for the same reason.

Findings so far (see the generated report for the numbers):

· `--min-suffix-support` — `C3` and `generic` do **not** use it, so the headline "how much of C3
  is free noise" is threshold-free. The `affix` column does use it, and `affix ÷ generic` crosses
  1.0 between the ends of the sweep, so an affix share is meaningless without its threshold.
· `--helper-lines` — stable to about ±1 percentage point over a 20x range.
· `--split` — stable to 1–2 points.
· `--alloc-frac` — **not stable for the monolingual zero-point** (V1 drifts monotonically to
  0.61 as the sample grows), which is why C5's default is the whole corpus.

Usage:
    python sensitivity.py                 # the full sweep (~15-20 min)
    python sensitivity.py --quick         # a reduced grid, for a smoke test
    python sensitivity.py --keep-run-dir  # do not delete the per-config outputs

Outputs:
    out/sensitivity.md   every swept quantity, with a verdict per setting
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COST = Path(__file__).resolve().parent / "cost_analysis.py"

# The variants that carry the conclusions. Sweeping all 14 would multiply the runtime by three
# and add nothing: these are the zero-point (V1), the main experiment (V3), its recovery arms
# (V4/R1, V5/R2) — in both segmentation models where both exist.
KEY_VARIANTS = ["V1_az_8000_unigram", "V3_8000_unigram", "V3_8000_bpe",
                "V4_8000_bpe_a0.2", "V5_8000_unigram"]

# A cheap allocation sample for the sweeps that do not concern C5. Using it keeps those runs to
# seconds instead of minutes — but it means **C5 must not be read from them**, which is why
# `alloc` is the one sweep that runs the full corpora.
FAST_ALLOC = "0.02"


def run_cost(out_dir: Path, extra: list[str], forward: list[str]) -> dict[tuple[str, str], dict]:
    """Run the real cost_analysis.py. `forward` carries --data-dir/--vocab-dir when the artifacts
    do not sit next to the repository root (they never do — `out/` is gitignored, so a fresh clone
    has `out/vocabs/index.json` and nothing else)."""
    cmd = [sys.executable, str(COST),
           "--out-dir", str(out_dir),
           "--variants", *KEY_VARIANTS,
           *forward,
           *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"cost_analysis.py failed for {extra}:\n{proc.stderr[-2000:]}")
    with open(out_dir / "cost_metrics.csv", encoding="utf-8") as f:
        return {(r["tag"], r["lang"]): r for r in csv.DictReader(f)}


def num(row: dict, key: str):
    v = (row or {}).get(key)
    return float(v) if v not in (None, "", "None") else None


def headline(rows: dict, include_c5: bool) -> dict:
    """The quantities the report actually quotes, extracted for one configuration.

    ⚠️ `include_c5` is False for every sweep except the one that actually varies the allocation
    sample. Those sweeps run with a deliberately cheap `--alloc-frac` to keep them to seconds, so
    their C5 values are **not readable** — and printing them anyway would put numbers in the
    report that nothing should quote. Leave them out rather than label them.
    """
    az = rows.get(("V3_8000_unigram", "az"))
    v1 = rows.get(("V1_az_8000_unigram", "az"))
    v4 = rows.get(("V4_8000_bpe_a0.2", "az"))
    v5 = rows.get(("V5_8000_unigram", "az"))

    def split(r):
        c3 = num(r, "C3_share_tr")
        gen = None
        if r is not None:
            frag, junk = num(r, "C6_fragment_share_of_shared"), num(r, "C6_junk_share_of_shared")
            gen = (frag or 0) + (junk or 0)
        return c3, gen

    c3, gen = split(az)
    _, gen4 = split(v4)
    _, gen5 = split(v5)
    out = {
        "C3 (V3 az)": c3,
        "generic ÷ C3 (V3 az)": (gen / c3) if (c3 and gen is not None) else None,
        "morph = C3 − generic (V3 az)": (c3 - gen) if (c3 and gen is not None) else None,
        "generic ÷ C3 (V4 α=0.2)": (gen4 / num(v4, "C3_share_tr")) if (v4 and gen4 is not None) else None,
        "generic ÷ C3 (V5)": (gen5 / num(v5, "C3_share_tr")) if (v5 and gen5 is not None) else None,
        "affix ÷ generic (V3 az)": (num(az, "C6_useful_share_of_shared") / gen) if (az and gen) else None,
    }
    if include_c5:
        out["C5 alloc (V1 az) — the monolingual zero-point"] = num(v1, "C5_alloc_share")
        out["C5 alloc (V3 az)"] = num(az, "C5_alloc_share")
    return out


# (title, why it matters, [(label, extra flags)])
SWEEPS = [
    ("`--min-suffix-support` — what counts as a productive affix",
     "C6's `affix` column uses it; C3 and `generic` do not. If a conclusion moves here, it is a "
     "statement about the threshold, not about the data.",
     [("5", ["--min-suffix-support", "5", "--alloc-frac", FAST_ALLOC]),
      ("20 (default)", ["--min-suffix-support", "20", "--alloc-frac", FAST_ALLOC]),
      ("100", ["--min-suffix-support", "100", "--alloc-frac", FAST_ALLOC])]),

    ("`--helper-lines` — how much helper text defines the shared-piece set",
     "C3's denominator is 'which pieces the helper language has used'. A small sample would "
     "understate it.",
     [("10,000", ["--helper-lines", "10000", "--alloc-frac", FAST_ALLOC]),
      ("50,000 (default)", ["--helper-lines", "50000", "--alloc-frac", FAST_ALLOC]),
      ("200,000", ["--helper-lines", "200000", "--alloc-frac", FAST_ALLOC])]),

    ("`--split` — which evaluation split the metrics are computed on",
     "If a conclusion only holds on one split, it is not a conclusion.",
     [("dev", ["--split", "dev", "--alloc-frac", FAST_ALLOC]),
      ("test (default)", ["--split", "test", "--alloc-frac", FAST_ALLOC])]),

    ("`--alloc-frac` — how much of each corpus C5 samples",
     "⚠️ C5 assigns every slot to the language that uses it most *in the sample*, so this decides "
     "the answer. Unlike the others this sweep runs the full corpora at the top end.",
     [("5%", ["--alloc-frac", "0.05"]),
      ("10%", ["--alloc-frac", "0.10"]),
      ("25%", ["--alloc-frac", "0.25"]),
      ("100% (default)", ["--alloc-frac", "1.0"])]),
]

# Quantities that move **by construction**: the setting is part of their definition, so a sweep is
# *expected* to move them and the actionable finding is only "quote the setting". These must not be
# reported in the same breath as a quantity that moves because the protocol changes the answer —
# one is a labelling duty, the other means the number was never well defined.
DEFINITIONAL = {"affix ÷ generic (V3 az)": "the threshold *defines* what 'affix' means"}

QUICK = {"`--min-suffix-support` — what counts as a productive affix": [0, 1],
         "`--helper-lines` — how much helper text defines the shared-piece set": [1],
         "`--split` — which evaluation split the metrics are computed on": [1],
         "`--alloc-frac` — how much of each corpus C5 samples": [3]}


def fmt(v, pct: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out")
    ap.add_argument("--data-dir", type=Path, default=None, help="forwarded to cost_analysis.py")
    ap.add_argument("--vocab-dir", type=Path, default=None, help="forwarded to cost_analysis.py")
    ap.add_argument("--quick", action="store_true", help="run a reduced grid as a smoke test")
    ap.add_argument("--keep-run-dir", action="store_true", help="do not delete the per-config outputs")
    args = ap.parse_args()

    forward: list[str] = []
    for flag, val in (("--data-dir", args.data_dir), ("--vocab-dir", args.vocab_dir)):
        if val is not None:
            forward += [flag, str(val)]

    tmp = Path(tempfile.mkdtemp(prefix="wp0-sens-"))
    results: list[tuple[str, str, list[tuple[str, dict]]]] = []
    try:
        for title, why, configs in SWEEPS:
            if args.quick and title in QUICK:
                configs = [configs[i] for i in QUICK[title] if i < len(configs)]
            # C5 is only meaningful in the sweep that varies the allocation sample — the others
            # run with a deliberately cheap sample, so their C5 values must not be shown.
            include_c5 = "--alloc-frac" in title
            rows = []
            for label, extra in configs:
                print(f"  {title.split('—')[0].strip():24s} {label:>18s} …", flush=True)
                out = tmp / f"{abs(hash((title, label)))}"
                out.mkdir(parents=True, exist_ok=True)
                rows.append((label, headline(run_cost(out, extra, forward), include_c5)))
            results.append((title, why, rows))
    finally:
        if not args.keep_run_dir:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"\nper-config outputs kept in {tmp}")

    # ---------------------------------------------------------------- report
    md = ["# WP0 · Sensitivity of the analysis settings", "",
          "Every headline number rests on someone's choice of setting. This runs the **real**",
          "`cost_analysis.py` under varied flags and records whether each conclusion moves.",
          "A quantity that moves here is a statement about the setting, not about the data.", "",
          f"Variants swept: {', '.join('`' + v + '`' for v in KEY_VARIANTS)}.", ""]
    verdicts = []
    for title, why, rows in results:
        keys = list(rows[0][1].keys())
        md += [f"## {title}", "", why, "",
               "| Setting | " + " | ".join(keys) + " |",
               "|---" * (len(keys) + 1) + "|"]
        for label, vals in rows:
            md.append(f"| {label} | " + " | ".join(
                fmt(vals[k], pct="÷" in k or "÷ C3" in k) for k in keys) + " |")
        # verdict: for each quantity, did it move materially across the sweep?
        md += [""]
        for k in keys:
            vs = [v[k] for _, v in rows if v[k] is not None]
            if len(vs) < 2:
                continue
            lo, hi = min(vs), max(vs)
            ratio = (hi / lo) if lo else float("inf")
            if ratio > 1.25:
                verdicts.append((title, k, fmt(lo), fmt(hi), ratio))
                md += [f"- `{k}`: **MOVES** across the sweep — {fmt(lo)} → {fmt(hi)} "
                       f"({ratio:.2f}x). Do not quote a single value without its setting."]
            else:
                md += [f"- `{k}`: stable ({fmt(lo)} → {fmt(hi)})."]
        md += [""]        # blank line, or the next "## " heading glues onto the bullet list

    md += ["", "## Verdict", ""]
    by_construction = [v for v in verdicts if v[1] in DEFINITIONAL]
    as_measurement = [v for v in verdicts if v[1] not in DEFINITIONAL]

    if as_measurement:
        md += ["**These moved as a *measurement* — the protocol changed the answer.** This is the",
               "dangerous kind: the number was never well defined, and a value quoted from one",
               "setting is not a fact about the data.", ""]
        for title, k, lo, hi, ratio in as_measurement:
            md.append(f"- `{k}` — {lo} → {hi} ({ratio:.2f}x)   under {title.split('—')[0].strip()}")
        md += [""]
    if by_construction:
        md += ["**These moved *by construction* — the setting is part of their definition.** No",
               "surprise here; the finding is a labelling duty, not a warning about the data.", ""]
        for title, k, lo, hi, ratio in by_construction:
            md.append(f"- `{k}` — {lo} → {hi} ({ratio:.2f}x): {DEFINITIONAL[k]}"
                      f"   ({title.split('—')[0].strip()})")
        md += [""]
    if not verdicts:
        md += ["Every swept quantity is stable across the settings tried."]
    else:
        md += ["Anything in either list that reaches the report must carry its setting, and where",
               "the quantity sits in the first list, the range should be quoted with it."]
    md += ["", "> Scope note: this sweeps the *analysis* settings. The upstream choices —",
           "> vocabulary size (8k only), the language set, the `HELPER` mapping — are separate,",
           "> and `directionality.py` already tests the `HELPER` mapping for az.", ""]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n→ {args.out_dir / 'sensitivity.md'}")
    print(f"   {len(verdicts)} quantity/quantities flagged as setting-dependent" if verdicts
          else "   all swept quantities stable")


if __name__ == "__main__":
    main()
