# MNLP project 2026 Fall

## Quick start

For step-by-step download, extraction and preprocessing instructions, see the
[data preparation guide](data/README.md).

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

```text
MNLP_Project5/
|-- notebooks/                 # Experiment workflows and data exploration
|-- src/
|   |-- data.py                # Load aligned language pairs
|   |-- model.py               # Basic translation model
|   |-- evaluation.py          # BLEU, chrF and COMET evaluation
|   `-- preprocessing/         # Data preparation, vocabulary training and analyses
|-- data/
|   |-- TED.tar                # Downloaded corpus
|   |-- raw/                   # Extracted corpus with alignment preserved
|   `-- processed/
|       |-- parallel/          # Aligned pairs and tracked manifest.json
|       |-- pivot/             # Aligned triples
|       |-- vocab_training/    # Non-empty text for vocabulary training only
|       |-- tokenized/         # Tokenized corpora
|       `-- nllb/              # NLLB notebook train/dev/test JSONL files
|-- artifacts/
|   |-- vocabularies/          # Tokenizers and tracked index.json
|   `-- checkpoints/           # Saved model checkpoints
|-- results/
|   |-- translations/<system>/ # Matching .src, .ref and .hyp files
|   |-- metrics/<system>/      # Scores, run metadata and training logs
|   |-- metrics/preprocessing/ # Vocabulary/corpus analysis reports
|   `-- figures/<system>/      # Plots
|-- tests/                     # Code checks
`-- Makefile                   # Preprocessing pipeline
```

`make all` runs preprocessing, not model training. Generated data, artifacts and
results are ignored by Git; the parallel manifest and vocabulary index are tracked.
`make clean` removes generated preprocessing data and tokenizers while preserving
those two metadata files, the downloaded archive, checkpoints and results.

### Start exploring

Open notebooks from the repository root or its `notebooks/` folder. Their setup
cells find the repository root, so output paths do not depend on the notebook's
working directory. The NLLB notebook requires a project checkout, including when
used in Colab. Start with
[data overview](notebooks/data_overview.ipynb) for the corpus and
[NLLB baselines](notebooks/baselines_nllb.ipynb)
for translation experiments.

Evaluate saved translations with, for example:

```python
from src.evaluation import evaluate_all_directions
scores = evaluate_all_directions("results/translations/nllb_basemodel", "bleu")
```

## Running the experiments

Everything below assumes the data is prepared (`make all`, or `make data corpora` for just the
aligned corpora the models need). Directions evaluated: `az-en`, `en-az`, `be-en`, `en-be`.
Each run is named `<name>_s<seed>` and writes to `results/translations/<run>/` (`.src`, `.ref`, `.hyp`
per direction) and `results/metrics/<run>/scores.json` (BLEU, chrF, COMET with per-sentence COMET).

### Systems

| System | Script | What it does |
|---|---|---|
| LSTM direct | `scripts/lstm.py` | Seq2seq LSTM trained on the target pair only |
| LSTM pivot | `scripts/lstm.py --pivot P` | Two LSTMs: src→P and P→tgt (`P` = `tr`, `uk`, `ru`) |
| NLLB base | `scripts/nllb.py --base` | Zero-shot `nllb-200-distilled-600M`, no training |
| NLLB pivot | `scripts/nllb.py --pivot P` | Zero-shot, translates src→P→tgt with the same pretrained model. **No training happens**, so seeds give identical output |
| NLLB fine-tune | `scripts/nllb.py --helpers ...` | Fine-tunes NLLB on az/be, optionally mixed with helper-language data (`tr`, `uk`, `ru`) |

### Locally

```bash
uv run python scripts/lstm.py --name lstm_direct --directions az-en en-az be-en en-be --seed 42
uv run python scripts/lstm.py --name lstm_pivot_tr --pivot tr --directions az-en en-az --seed 42
uv run python scripts/nllb.py --name nllb_basemodel --base
uv run python scripts/nllb.py --name nllb_pivot_tr --pivot tr --seed 42
uv run python scripts/nllb.py --name nllb_helpers_tr --helpers tr --seed 42
```

NLLB scripts need a CUDA GPU. Useful flags:

- `lstm.py`: `--arch {attn,basic}` (default `attn`: 50 epochs, max length 100; `basic`: 10 epochs, max length 20;
  non-`basic` runs get an `_attn` suffix in the run name), `--epochs`, `--max-length`, `--vocab-size`,
  `--max-pairs` (cap training pairs, handy for the huge pivot corpora).
- `nllb.py`: `--helpers tr uk ru`, `--alpha` (temperature sampling over languages, default 0.2), `--epochs`,
  `--lr`, `--direct tr-az az-tr` (extra direct pairs), `--match-budget` (same number of training examples as
  the target-only run), `--select-target` (pick the best epoch on az/be dev only), `--stage2-epochs N`
  (polish on az/be afterwards).

### On Snellius (Slurm)

Compute nodes have no internet, so set up once on a **login node**:

```bash
bash slurm/setup.sh      # installs uv, syncs env, prepares data, caches NLLB + COMET models
bash slurm/run_all.sh    # submits base, LSTM direct/pivot, NLLB fine-tune/pivot for seeds 42-44
bash slurm/run_helper_study.sh   # ablations on why Turkish does or does not help Azerbaijani
```

Job templates live in `slurm/*.sbatch` (A100 partition `gpu_a100`); logs go to `logs/`.
Submit a single job with e.g.
`sbatch --export=ALL,SEED=42,NAME=nllb_helpers_tr,HELPERS=tr slurm/nllb_finetune.sbatch` (join several
helpers with `_`, e.g. `HELPERS=tr_uk_ru`; extra flags after the script name are passed through).

An empty folder under `results/metrics/` means the job was created but never finished: check `logs/`.

### RQ1 helper-language study

`scripts/rq1_train.py` (fine-tune with one helper: `--run {none,uk,ru_matched,ru,tr}`, `--smoke` for a quick
check) and its companions: `check_rq1.py` (are the runs healthy?), `comet_rq1.py` (COMET from saved
translations), `significance_rq1.py` (paired bootstrap), `language_features_rq1.py` (language distance vs gain).
Cluster job files are in `scripts/snellius/`. See each script's docstring for options.

### Evaluating and tests

```bash
uv run python -c "from src.evaluation import evaluate_all_directions; print(evaluate_all_directions('results/translations/nllb_basemodel_s42', 'chrf'))"
uv run python -m pytest tests          # or: python -m unittest discover -s tests -v
```
