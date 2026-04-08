#!/bin/bash
# Exp 0: AlphaGenome S2 yeast scaling — EXTENDED fractions.
#
# Adds the missing fractions to complete the full scaling curve:
#   0.002 (0.2%), 0.10 (10%), 0.20 (20%), 0.50 (50%), 1.00 (100%)
#
# The low fractions (0.001, 0.005, 0.01, 0.02, 0.05) are already running
# in job 836669 (exp0_yeast_scaling_ag_s2.sh).
#
# High fractions are exploratory — S2 fine-tuning on millions of sequences
# is expensive, so we use generous wall time to see how far they get.
# --requeue handles SLURM preemption: training script resumes from
# s2_progress.json + last_model_s2 checkpoint automatically.
#
# 5 fractions × 3 seeds = 15 tasks.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_yeast_scaling_ag_s2_extended.sh
#
#SBATCH --job-name=exp0_ag_yeast_s2x
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
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


# ~6M train sequences total
# Fractions: 0.002, 0.10, 0.20, 0.50, 1.00
MAX_SEQS=(12000 600000 1200000 3000000 6065325)
FRAC_LABELS=("0.002" "0.10" "0.20" "0.50" "1.00")
SEEDS=(42 123 456)
N_FRACS=${#MAX_SEQS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_FRACS ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % N_FRACS ))
MAX_SEQ=${MAX_SEQS[$FRAC_IDX]}
FRAC_LABEL=${FRAC_LABELS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Same output dir as the original script — fractions interleave cleanly
OUT_DIR="outputs/exp0_yeast_scaling_ag_s2/fraction_${FRAC_LABEL}/seed_${SEED}"

echo "=== AG yeast S2 scaling (extended): fraction=${FRAC_LABEL} (max_seq=${MAX_SEQ}) seed=${SEED} ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
START_TIME=$(date +%s)

# Skip if result already exists
if [ -f "${OUT_DIR}/summary.json" ]; then
    echo "SKIP: result already exists"
    exit 0
fi

# Note: max_wall_seconds set to 6 days (518400s) with fraction 0.80 → stops at ~4.8 days
# to leave time for test evaluation + checkpoint saving before 7-day wall time.
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
    "++eval_use_reverse_complement=false" \
    "++max_wall_seconds=518400"

# Clean up checkpoints to save disk space (only after successful completion)
rm -rf "${OUT_DIR}/best_model" "${OUT_DIR}/last_model" "${OUT_DIR}/stage1_best" "${OUT_DIR}/last_model_s2" "${OUT_DIR}/s2_progress.json" 2>/dev/null
echo "Cleaned up checkpoint dirs"

END_TIME=$(date +%s)
echo "=== fraction=${FRAC_LABEL} seed=${SEED} DONE — wall time: $((END_TIME - START_TIME))s ==="
