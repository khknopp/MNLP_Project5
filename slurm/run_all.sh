#!/bin/bash
cd "$(dirname "$0")/.."
mkdir -p logs

prep=$(sbatch --parsable slurm/prep.sbatch)
after="--dependency=afterok:${prep}"

sbatch $after slurm/lstm_direct.sbatch
sbatch $after slurm/nllb_base.sbatch
sbatch $after --export=ALL,NAME=nllb_finetuned_no_helpers,HELPERS= slurm/nllb_finetune.sbatch
sbatch $after --export=ALL,NAME=nllb_helpers_tr,HELPERS=tr slurm/nllb_finetune.sbatch
sbatch $after --export=ALL,NAME=nllb_helpers_uk,HELPERS=uk slurm/nllb_finetune.sbatch
sbatch $after --export=ALL,NAME=nllb_helpers_ru,HELPERS=ru slurm/nllb_finetune.sbatch
sbatch $after --export=ALL,NAME=nllb_helpers_all,HELPERS="tr uk ru" slurm/nllb_finetune.sbatch

for p in tr uk ru; do
    sbatch $after --export=ALL,PIVOT=$p slurm/lstm_pivot.sbatch
    sbatch $after --export=ALL,PIVOT=$p slurm/nllb_pivot.sbatch
done
