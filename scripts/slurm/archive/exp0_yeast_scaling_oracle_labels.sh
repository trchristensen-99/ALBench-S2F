#!/bin/bash
#SBATCH --job-name=exp0_yeast_oracle
#SBATCH --output=logs/exp0_yeast_oracle_%a.out
#SBATCH --error=logs/exp0_yeast_oracle_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --array=0-29

set -euo pipefail

# Source system profiles for modules and tmpdir
set +u
source /etc/profile.d/modules.sh
set -u
module load EB5

cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
source scripts/slurm/setup_hpc_deps.sh

OUTPUT_DIR="${OUTPUT_DIR:-outputs/exp0_yeast_scaling_oracle_labels}"
PSEUDOLABEL_DIR="${PSEUDOLABEL_DIR:-outputs/oracle_pseudolabels/yeast_dream_oracle}"
mkdir -p "${OUTPUT_DIR}"

# Map array index to (replicate_slot, fraction).
FRACTIONS=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)
REPLICATES=(0 1 2)
N_FRAC=${#FRACTIONS[@]}
FRAC_IDX=$((SLURM_ARRAY_TASK_ID % N_FRAC))
REP_IDX=$((SLURM_ARRAY_TASK_ID / N_FRAC))
FRACTION=${FRACTIONS[$FRAC_IDX]}
REPLICATE_SLOT=${REPLICATES[$REP_IDX]}
FRAC_FMT=$(printf "%.4f" "${FRACTION}")

echo "=== Exp 0 Yeast Oracle-Label Scaling: replicate_slot=${REPLICATE_SLOT} fraction=${FRACTION} task=${SLURM_ARRAY_TASK_ID} ==="

# Reuse existing completed runs for this fraction to avoid reruns.
EXISTING_COUNT=$(find "${OUTPUT_DIR}" -path "*/fraction_${FRAC_FMT}/result.json" 2>/dev/null | wc -l)
if (( EXISTING_COUNT > REPLICATE_SLOT )); then
    echo "Skipping fraction=${FRACTION}: already have ${EXISTING_COUNT} completed run(s), slot=${REPLICATE_SLOT}"
    exit 0
fi

# W&B auth
if [[ -f ~/.wandb_key ]]; then
    export WANDB_API_KEY=$(cat ~/.wandb_key)
elif [[ -f .env ]]; then
    export $(grep WANDB_API_KEY .env | xargs)
fi

uv run --no-sync python experiments/exp0_yeast_scaling_oracle_labels.py \
    fraction="${FRACTION}" \
    output_dir="${OUTPUT_DIR}" \
    pseudolabel_dir="${PSEUDOLABEL_DIR}" \
    seed=null \
    num_workers=4 \
    test_subset_dir=data/yeast/test_subset_ids \
    wandb_mode=offline
