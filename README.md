# WP0 · Preprocessing — the cost of a joint subword vocabulary, and whether it can be recovered

> **Start with `notebooks/data_overview.ipynb`.** It is the project's single document: the data
> and how to read it, the experiment design (V1–V5, C1–C6), the results, which numbers survive a
> setting change, the open decisions and the house rules. This README is **setup only**.
>
> Owner: Zixin · MNLP_Project5 · due 2026-10-09

---

## Quick start

**The corpus is not in the repository.** TED is CC-BY-NC-ND-4.0 and ND forbids distributing
derivative works — the unpacked text, the tokenized corpora and **even the trained
vocabularies** are all derivatives. So everyone downloads and runs it themselves.

```bash
uv sync         # creates .venv and installs the pinned tokenizers (see pyproject.toml)
make all        # download TED.tar + run the whole pipeline + verify
```

`make` detects uv automatically and runs everything through `uv run`, so the pinned
`tokenizers` version is always the one in effect. Without uv:

```bash
pip install -r requirements.txt
make all        # falls back to plain python3
```

Single steps:

```bash
make data       # TED.tar -> data/raw/ (blank lines kept) + data/processed/ (blank lines dropped)
make corpora    # -> out/parallel/ + manifest.json, and out/pivot/
make vocabs     # 16 vocabulary variants (unigram + bpe), each with a recorded sha256
make cost       # -> out/cost_metrics.csv / out/cost_report.md (C1-C6)
make directionality  # -> out/directionality_report.md + out/directionality.csv (translation-corpus evidence, corrected C3)
make sensitivity     # -> out/sensitivity.md (~15-20 min; deliberately not part of `all`)
make verify     # two checks: corpora vs manifest.json, vocabularies vs index.json
make check      # alignment integrity + leakage report
```

Adding a variant to an existing `out/vocabs/` without retraining the rest:

```bash
python preprocessing/train_vocabs.py --rehash   # record/refresh hashes only; trains nothing
```

**Before any downstream experiment, `make verify` must be green.** Why the manifest and the
`tokenizers` pin are what make five people's numbers comparable, and what to do when you get
artifacts from a teammate instead of running the pipeline: `data_overview.ipynb`, "Where the
data comes from, and how every number is reproduced".

---

## Directory

```
MNLP_Project5/
├── README.md              <- this file: setup only
├── Makefile               <- one command end to end: make all / make verify / make directionality
├── requirements.txt       <- full project environment; mirrors pyproject.toml
├── pyproject.toml         <- the same dependencies for the uv path; pins tokenizers exactly
├── preprocessing/         <- WP0 scripts — what each does: data_overview.ipynb, "Where everything lives"
├── src/                   <- model code (Kai): data.py, model.py
├── notebooks/
│   ├── data_overview.ipynb                    <- START HERE: the data, the design, the results, the open decisions
│   ├── test_basic_model.ipynb                 <- Kai's smoke test
│   └── baselines_2_3_off_the_shelf_and_wo_helper.ipynb   <- Pinar's NLLB baselines
├── data/                  <- TED.tar + raw/ + processed/ (not committed, CC-BY-NC-ND)
└── out/                   <- artifacts; only parallel/manifest.json and vocabs/index.json are committed
```

> Every script resolves `data/` and `out/` relative to the **repository root**, so `make …` and
> `python preprocessing/<script>.py` both work from anywhere. `data/raw/` keeps blank lines and is
> the only source for parallel corpora; `data/processed/` drops them and feeds vocabulary training
> only — what breaks silently if you mix them up: `data_overview.ipynb` §2 and §5.
