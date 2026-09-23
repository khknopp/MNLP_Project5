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
[NLLB baselines](notebooks/baselines_2_3_off_the_shelf_and_wo_helper.ipynb)
for translation experiments.

Evaluate saved translations with, for example:

```python
from src.evaluation import evaluate_all_directions
scores = evaluate_all_directions("results/translations/nllb_basemodel", "bleu")
```
