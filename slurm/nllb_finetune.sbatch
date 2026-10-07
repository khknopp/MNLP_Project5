#!/bin/bash
#SBATCH --job-name=nllb_finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00

source slurm/env.sh
uv run python scripts/nllb.py --name ${NAME} --helpers ${HELPERS//_/ } --alpha ${ALPHA:-0.2} --seed ${SEED} "$@"
