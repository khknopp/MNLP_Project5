#!/bin/bash
# One-time setup. Run on a Snellius LOGIN node (needs internet):  bash slurm/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

# uv (user-local install; also fetches the pinned Python 3.12.11 from .python-version)
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen

# Data: TED.tar download, extraction, aligned corpora (idempotent)
make data corpora
uv run python -c "from src.data import TedData; TedData()"
uv run python src/preprocessing/verify.py || echo "WARNING: verify.py reported differences"

# Pre-download models so offline compute nodes can load them
uv run python - <<'PY'
from huggingface_hub import snapshot_download
from comet import download_model, load_from_checkpoint
snapshot_download("facebook/nllb-200-distilled-600M")
# load_from_checkpoint also fetches COMET's xlm-roberta-large tokenizer/config, which offline jobs need
load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
PY
touch .cache/setup_done
echo "Setup done. Submit with: bash slurm/run_all.sh"
