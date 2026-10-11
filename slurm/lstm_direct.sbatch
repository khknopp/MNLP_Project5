#!/bin/bash
#SBATCH --job-name=lstm_direct
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00

source slurm/env.sh
uv run python scripts/lstm.py --name lstm_direct --directions az-en en-az be-en en-be --seed ${SEED} "$@"
