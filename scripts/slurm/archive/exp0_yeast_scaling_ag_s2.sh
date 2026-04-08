#!/bin/bash
# Exp 0: AlphaGenome S2 (encoder fine-tuning) scaling curve on yeast.
#
# Lower fractions only (S2 is expensive: ~8-12h per fraction).
# S1 trains on all data (cached, fast); S2 fine-tunes at each fraction.
# 3 seeds x 5 fractions = 15 tasks.
#
# Uses best config from v5 sweep (update if v5 produces a different winner).
# Current config: encoder unfreezing, lr=5e-4, wd=1e-6.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_yeast_scaling_ag_s2.sh
#
#SBATCH --job-name=exp0_ag_yeast_s2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-14

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh


# ~6M train sequences; these correspond to fractions 0.001, 0.005, 0.01, 0.02, 0.05
MAX_SEQS=(6000 30000 60000 120000 300000)
FRAC_LABELS=("0.001" "0.005" "0.01" "0.02" "0.05")
SEEDS=(42 123 456)
N_FRACS=${#MAX_SEQS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_FRACS ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % N_FRACS ))
MAX_SEQ=${MAX_SEQS[$FRAC_IDX]}
FRAC_LABEL=${FRAC_LABELS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT_DIR="outputs/exp0_yeast_scaling_ag_s2/fraction_${FRAC_LABEL}/seed_${SEED}"

echo "=== AG yeast S2 scaling: fraction=${FRAC_LABEL} (max_seq=${MAX_SEQ}) seed=${SEED} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if result already exists
if [ -f "${OUT_DIR}/summary.json" ]; then
    echo "SKIP: result already exists"
    exit 0
fi

uv run --no-sync python experiments/train_oracle_alphagenome_yeast.py \
    --config-name oracle_alphagenome_yeast_finetune_sweep \
    "++output_dir=${OUT_DIR}" \
    "++seed=${SEED}" \
    "++wandb_mode=offline" \
    "++batch_size=4096" \
    "++lr=0.003" \
    "++epochs=5" \
    "++early_stop_patience=100" \
    "++second_stage_epochs=50" \
    "++second_stage_batch_size=1024" \
    "++second_stage_early_stop_patience=7" \
    "++second_stage_lr=5e-4" \
    "++second_stage_weight_decay=1e-6" \
    "++second_stage_max_sequences=${MAX_SEQ}" \
    "++second_stage_unfreeze_mode=encoder" \
    "++eval_use_reverse_complement=false"

# Clean up checkpoints to save disk space
rm -rf "${OUT_DIR}/best_model" "${OUT_DIR}/last_model" "${OUT_DIR}/stage1_best" "${OUT_DIR}/last_model_s2" 2>/dev/null
echo "Cleaned up checkpoint dirs"

END_TIME=$(date +%s)
echo "=== fraction=${FRAC_LABEL} seed=${SEED} DONE — wall time: $((END_TIME - START_TIME))s ==="
