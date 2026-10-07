#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs

# Compute nodes have no internet, so the environment, data and models are prepared here on the login node.
[ -f .cache/setup_done ] || bash slurm/setup.sh
after=""

sbatch $after slurm/nllb_base.sbatch

for seed in 42 43 44; do
    sbatch $after --export=ALL,SEED=$seed slurm/lstm_direct.sbatch
    sbatch $after --export=ALL,SEED=$seed,NAME=nllb_finetuned_no_helpers,HELPERS= slurm/nllb_finetune.sbatch
    sbatch $after --export=ALL,SEED=$seed,NAME=nllb_helpers_tr,HELPERS=tr slurm/nllb_finetune.sbatch
    sbatch $after --export=ALL,SEED=$seed,NAME=nllb_helpers_uk,HELPERS=uk slurm/nllb_finetune.sbatch
    sbatch $after --export=ALL,SEED=$seed,NAME=nllb_helpers_ru,HELPERS=ru slurm/nllb_finetune.sbatch
    sbatch $after --export=ALL,SEED=$seed,NAME=nllb_helpers_all,HELPERS=tr_uk_ru slurm/nllb_finetune.sbatch

    for p in tr uk ru; do
        sbatch $after --export=ALL,SEED=$seed,PIVOT=$p slurm/lstm_pivot.sbatch
        sbatch $after --export=ALL,SEED=$seed,PIVOT=$p slurm/nllb_pivot.sbatch
    done
done
