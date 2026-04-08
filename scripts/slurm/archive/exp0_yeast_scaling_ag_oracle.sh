#!/bin/bash
# Exp 0: AlphaGenome cached-head scaling curve on yeast — ORACLE labels.
#
# 3 seeds × 10 fractions = 30 tasks.
# Uses frozen S1 encoder embeddings (cached) + oracle pseudolabels from DREAM-RNN ensemble.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_yeast_scaling_ag_oracle.sh
#
#SBATCH --job-name=exp0_ag_yeast_oracle
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --array=0-29

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh


FRACTIONS=(0.001 0.002 0.005 0.01 0.02 0.05 0.10 0.20 0.50 1.00)
SEEDS=(42 123 456)
N_FRACTIONS=${#FRACTIONS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_FRACTIONS ))
FRAC_IDX=$(( SLURM_ARRAY_TASK_ID % N_FRACTIONS ))
FRACTION=${FRACTIONS[$FRAC_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT_DIR="outputs/exp0_yeast_scaling_ag_oracle"

echo "=== AG yeast oracle scaling: fraction=${FRACTION} seed=${SEED} (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if result already exists
if [ -f "${OUT_DIR}/fraction_${FRACTION}/seed_${SEED}/result.json" ]; then
    echo "SKIP: result already exists"
    exit 0
fi

uv run --no-sync python experiments/exp0_yeast_scaling_alphagenome.py \
    ++fraction="${FRACTION}" \
    ++seed="${SEED}" \
    ++output_dir="${OUT_DIR}" \
    ++oracle_label_path=outputs/oracle_pseudolabels/yeast_dream_oracle \
    ++epochs=50 \
    ++early_stop_patience=7 \
    ++wandb_mode=offline \
    ++test_subset_dir=data/yeast/test_subset_ids

echo "=== DONE — $(date) ==="
