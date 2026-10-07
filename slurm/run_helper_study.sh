#!/bin/bash
# Why does Turkish not help Azerbaijani? Ablations on top of the existing nllb_helpers_tr / nllb_finetuned_no_helpers runs.
# Run after run_all.sh (or alone): bash slurm/run_helper_study.sh
#
# Suspected confounds in the original helper run (alpha=0.2, all helper data):
#   * it trains on ~386k examples vs ~21k for the no-helper run (18x more steps, az seen ~9x more often)
#   * the best epoch is picked on a dev set dominated by Turkish, not by az/be
#   * tr-en is 182k sentences, but the only direct tr<->az data is 5.6k pairs
# Flags (scripts/nllb.py):  --select-target  dev selection on az/be only
#                           --match-budget   same number of training examples as the no-helper run
#                           --direct tr-az az-tr   add direct Turkish<->Azerbaijani pairs
#                           --stage2-epochs  polish on az/be only afterwards
cd "$(dirname "$0")/.."
mkdir -p logs
[ -f .cache/setup_done ] || bash slurm/setup.sh

sub() { # name helpers [extra nllb.py flags...]
    local name=$1 helpers=$2; shift 2
    for seed in 42 43 44; do
        sbatch --export=ALL,SEED=$seed,NAME=$name,HELPERS=$helpers slurm/nllb_finetune.sbatch "$@"
    done
}

# H1: model selection on az/be dev only (everything else as the original tr run)
sub nllb_tr_seltarget          tr --select-target
# H2: equal compute and equal selection as the no-helper control: isolates the effect of the data itself
sub nllb_tr_fair               tr --select-target --match-budget
# H3: a different helper under identical fair conditions: is Turkish special?
sub nllb_ru_fair               ru --select-target --match-budget
# H4: direct tr<->az pairs only (closest relative, small but on-target data)
sub nllb_directtr_fair         "" --select-target --match-budget --direct tr-az az-tr
# H5: tr-en/en-tr plus the direct pairs
sub nllb_tr_directtr_fair      tr --select-target --match-budget --direct tr-az az-tr
# H6: full helper data, then polish on az/be only (helps without diluting the final model)
sub nllb_tr_stage2             tr --select-target --stage2-epochs 3
# H7: how much of the mixture should be helper? (fair budget; the original alpha=0.2 gives tr ~51%)
sub nllb_tr_fair_a005           tr --select-target --match-budget --alpha 0.05
sub nllb_tr_fair_a05           tr --select-target --match-budget --alpha 0.5
# H8: gentler fine-tuning, less forgetting of what NLLB already knows about tr/az
sub nllb_tr_fair_lr2e5         tr --select-target --match-budget --lr 2e-5
