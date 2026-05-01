#!/bin/bash
# Yeast Exp 0 — DREAM-RNN scaling on REAL labels (DREAM train + val,
# MAUDE test). Same 10-size × 3-rep grid as the oracle arm so the two
# scaling curves are directly comparable.
#
# Note: most of these points were already trained earlier (see
# outputs/exp0_yeast_dream_rnn_pilot/) — this script ensures we have a
# fresh, schema-consistent set that can be aggregated alongside the
# oracle-arm results. Existing identical fractions will be re-trained
# with the same HPs but different seeds.
#
#SBATCH --job-name=yeast_exp0_real
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-29

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
SIZES=(6065 12131 30327 60653 121307 303266 606532 1213065 3032662 6065324)
SEEDS=(42 1042 2042)
SIZE_IDX=$((T / 3))
REP_IDX=$((T % 3))
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$REP_IDX]}
FRACTION=$(awk "BEGIN { printf \"%.6f\", ${SIZE}/6065324 }")

OUT="outputs/exp0_yeast_real_scaling/random/n${SIZE}/rep${REP_IDX}"
echo "=== task=${T} size=${SIZE} fraction=${FRACTION} seed=${SEED} ==="

uv run --no-sync python experiments/exp0_yeast_scaling.py \
    fraction=${FRACTION} \
    seed=${SEED} \
    output_dir="${OUT}" \
    lr=0.005 \
    lr_lstm=0.005 \
    hidden_dim=320 \
    cnn_filters=160 \
    epochs=80 \
    batch_size=512 \
    dropout_lstm=0.3 \
    dropout_cnn=0.2 \
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
