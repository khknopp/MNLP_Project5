#!/bin/bash
#SBATCH --job-name=lstm_direct
#SBATCH --output=logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

cd "${SLURM_SUBMIT_DIR}"
uv run python scripts/lstm.py --name lstm_direct --directions az-en en-az be-en en-be --seed ${SEED}
