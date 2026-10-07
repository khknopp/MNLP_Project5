#!/bin/bash
#SBATCH --job-name=lstm_pivot
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

source slurm/env.sh
uv run python scripts/lstm.py --name lstm_pivot_${PIVOT} --pivot ${PIVOT} --directions az-en en-az be-en en-be --seed ${SEED}
