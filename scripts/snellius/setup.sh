#!/bin/bash
# One-time setup on Snellius. Run on a LOGIN node (compute nodes may have no internet):
#
#     cd ~/MNLP_Project5 && bash scripts/snellius/setup.sh
#
# Creates .venv, installs the project's packages, downloads TED.tar + extracts it, and caches the
# NLLB model, so the GPU job needs no network.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs   # SLURM writes the job logs here and does not create it

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0

[ -d .venv ] || python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt accelerate

# corpus: download + extract + check against the team's reference
python - <<'EOF'
from src.data import TedData
TedData()
EOF
# keep the team's manifest: emit rewrites it, then check the new files against the kept copy
[ -f data/reference_manifest.json ] || cp data/processed/parallel/manifest.json data/reference_manifest.json
python src/preprocessing/align.py emit
cp data/reference_manifest.json data/processed/parallel/manifest.json
python src/preprocessing/verify.py --manifest data/reference_manifest.json --dir data/processed/parallel

# model: into the Hugging Face cache, so the job can run offline
python - <<'EOF'
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
EOF

python -m unittest tests.test_helper_training
echo "Setup done. Smoke test: sbatch --array=0 scripts/snellius/rq1.job --smoke"
