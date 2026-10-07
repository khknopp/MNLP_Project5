#!/bin/bash
#SBATCH --job-name=nllb_pivot
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00

source slurm/env.sh
uv run python scripts/nllb.py --name nllb_pivot_${PIVOT} --pivot ${PIVOT} --seed ${SEED}
