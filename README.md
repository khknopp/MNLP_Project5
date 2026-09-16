# MNLP project 2026 Fall

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
