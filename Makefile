# WP0 preprocessing pipeline — one command end to end.
#
#   make all            download TED.tar + run everything + verify   <-- newcomers: just this
#   make verify         check artifacts against manifest.json + index.json (how the team stays in sync)
#   make check          alignment integrity + leakage report (numbers go into the report's Limitations)
#   make directionality the corpus is a translation corpus — evidence + the corrected C3
#   make clean          remove artifacts (keeps TED.tar, so you don't re-download 673MB)
#
# The corpus is never committed (CC-BY-NC-ND-4.0, see .gitignore). Everyone
# downloads and runs it themselves, then `make verify` confirms we are all
# working on the same data.
#
# Scripts live in src/preprocessing/ and resolve data/, artifacts/ and results/ relative to the *repository root*,
# so every target works no matter which directory make is invoked from.

# Prefer uv when it is available: pyproject.toml pins the exact tokenizers version the
# published numbers came from (see the comment there), and `uv run` honours it.
# Falls back to plain python3 so `make` still works for anyone without uv.
PY ?= $(shell command -v uv >/dev/null 2>&1 && echo "uv run python" || echo python3)
PREP := src/preprocessing
TAR_URL := https://huggingface.co/datasets/davidstap/ted_talks/resolve/main/data/TED.tar

.PHONY: all data corpora vocabs cost directionality sensitivity verify check clean

all: data corpora vocabs cost directionality verify

# ---------------------------------------------------------------- data
data: data/TED.tar
	$(PY) $(PREP)/align.py extract   # -> data/raw/       keeps blank lines (alignment contract)
	$(PY) $(PREP)/prep_data.py       # -> data/processed/vocab_training/ drops blank lines, vocab training only

data/TED.tar:
	@mkdir -p data
	curl -L --retry 3 -C - -o $@ "$(TAR_URL)"

# ---------------------------------------------------------------- parallel corpora
corpora: data
	$(PY) $(PREP)/align.py emit      # -> data/processed/parallel/ + manifest.json
	$(PY) $(PREP)/align.py pivot     # -> data/processed/pivot/

# ---------------------------------------------------------------- vocabularies
# Train both segmentation models: the C2 baseline must pair (language, model),
# so mixing Unigram and BPE zero-points makes the inflation numbers meaningless.
vocabs: data
	$(PY) $(PREP)/train_vocabs.py --model unigram
	$(PY) $(PREP)/train_vocabs.py --model bpe

# ---------------------------------------------------------------- analysis
cost: vocabs
	$(PY) $(PREP)/cost_analysis.py   # -> results/metrics/preprocessing/cost_metrics.csv / results/metrics/preprocessing/cost_report.md (C1-C6)

# Needs the vocabularies, and cross-checks its C3 against cost_metrics.csv so the two
# scripts cannot drift apart unnoticed.
directionality: cost
	$(PY) $(PREP)/directionality.py  # -> results/metrics/preprocessing/directionality_report.md + results/metrics/preprocessing/directionality.csv

# Runs the real cost_analysis.py under varied flags and records which conclusions move.
# Deliberately not part of `all`: it takes ~15-20 minutes, and it answers a question you
# only ask when a number is about to go into the report.
sensitivity: cost
	$(PY) $(PREP)/sensitivity.py     # -> results/metrics/preprocessing/sensitivity.md

# ---------------------------------------------------------------- checks
# Two checks, and both matter: the first is about the data, the second about the
# segmentation. Sharing one corpus does not stop two people from segmenting it
# differently — and that failure is silent, so it has to be checked, not assumed.
verify:
	$(PY) $(PREP)/verify.py
	$(PY) $(PREP)/verify.py --vocabs

check: data
	$(PY) $(PREP)/align.py check

clean:
	rm -rf data/raw data/processed/vocab_training data/processed/pivot data/processed/tokenized
	find data/processed/parallel -type f ! -name manifest.json -delete
	find artifacts/vocabularies -mindepth 1 -maxdepth 1 ! -name index.json -exec rm -rf {} +
