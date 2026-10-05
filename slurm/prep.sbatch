#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --output=logs/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

cd "${SLURM_SUBMIT_DIR}"
uv sync
uv run python -c "from src.data import TedData; TedData()"
