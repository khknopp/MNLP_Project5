#!/bin/bash
#SBATCH --job-name=nllb_base
#SBATCH --output=logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd "${SLURM_SUBMIT_DIR}"
uv run python scripts/nllb.py --name nllb_basemodel --base
