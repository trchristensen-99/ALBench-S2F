#!/bin/bash
# Yeast Exp 0 — LegNet scaling on ENSEMBLE ORACLE pseudolabels.
# 20-task array: 10 sizes × 2 reps. Uses cached pseudolabels from
# outputs/yeast_ensemble_oracle/ (5 DRNN + 5 DCNN ensemble).
#
#SBATCH --job-name=yeast_exp0_legoral
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-19

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
SIZES=(6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324)
SEEDS=(42 1042)
SIZE_IDX=$((T / 2))
REP_IDX=$((T % 2))
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$REP_IDX]}
FRACTION=$(awk "BEGIN { printf \"%.6f\", ${SIZE}/6065324 }")

OUT="outputs/exp0_yeast_legnet_oracle/random/n${SIZE}/rep${REP_IDX}"
echo "=== task=${T} size=${SIZE} fraction=${FRACTION} seed=${SEED} ==="

uv run --no-sync python experiments/exp0_yeast_oracle_scaling.py \
    +student=legnet \
    fraction=${FRACTION} \
    seed=${SEED} \
    output_dir="${OUT}" \
    lr=0.005 \
    lr_lstm=0.005 \
    epochs=80 \
    batch_size=512 \
    weight_decay=0.01 \
    use_reverse_complement=true \
    early_stopping_patience=10 \
    metric_for_best=pearson_r \
    use_amp=true \
    use_compile=false \
    pct_start=0.1 \
    num_workers=4 \
    pin_memory=true

echo "=== DONE — $(date) ==="
