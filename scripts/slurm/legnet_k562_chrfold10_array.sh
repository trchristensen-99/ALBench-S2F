#!/bin/bash
# 10-fold LegNet K562 chr-fold training, ON REAL K562_log2FC LABELS
# (for the bar-plot model comparison). One SLURM job per fold on slow_nice
# so all 10 train in parallel → wall ≈ 1 fold (~1.5h) vs ~5-6h serialized.
#
# HPs are the R4 winner from the prior AG-oracle HP sweep — they may not
# be optimal for real labels, but R4's lr=3e-4 / dense_dims=[256]*4 / bs=128
# is sensible.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

OUT_BASE=results/snv_eval/legnet_k562_chrfold10_real
mkdir -p "$OUT_BASE" "$REPO/logs"

JOBFILE=$(mktemp)
cat > "$JOBFILE" <<'EOF'
#!/bin/bash
#SBATCH --job-name=legnet_k562_chrfold_real
#SBATCH --output=__REPO__/logs/%x-%A_%a.out
#SBATCH --error=__REPO__/logs/%x-%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-9

cd __REPO__
source .venv/bin/activate
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export HP_FAST=1

VAL_CHRS=(1 2 3 4 5 6 8 9 10 11)
VAL_CHR=${VAL_CHRS[$SLURM_ARRAY_TASK_ID]}
SEED=$((42 + SLURM_ARRAY_TASK_ID))
OUT_DIR=__OUT_BASE__/fold${SLURM_ARRAY_TASK_ID}_val_chr${VAL_CHR}
mkdir -p "$OUT_DIR"

if [ -f "$OUT_DIR/result.json" ]; then
    echo "[skip fold $SLURM_ARRAY_TASK_ID] result.json already exists"
    exit 0
fi

echo "[legnet REAL labels fold $SLURM_ARRAY_TASK_ID] val_chr=$VAL_CHR  seed=$SEED  out=$OUT_DIR"

uv run --no-sync python -u scripts/preflight/run_single.py \
    --arch legnet --d_train 0 --seed $SEED \
    --epochs 60 --early_stop_patience 10 \
    --augmentations rev_complement \
    --label_source real \
    --val_chrs $VAL_CHR --test_chrs 7,13 \
    --output_dir $OUT_DIR \
    --fast \
    --hp lr=0.0003 batch_size=128 weight_decay=0 conv_dropout=0.1 dense_dropout=0 \
         block_sizes=[256,256,256,256] block_class=eff optimizer=adamw
EOF

sed -i.bak "s|__REPO__|$REPO|g; s|__OUT_BASE__|$OUT_BASE|g" "$JOBFILE"
rm -f "$JOBFILE.bak"

JID=$($SBATCH --parsable "$JOBFILE")
rm -f "$JOBFILE"
echo "LegNet K562 chr-fold REAL labels 10-array → $JID"
