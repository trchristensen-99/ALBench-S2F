#!/bin/bash
# Launch SCALING LAW chain for c91 DEBIASED oracle:
#   1. Wait for c91 10-fold training to complete (job IDs 2151798-2151801)
#   2. Generate c91 pool labels (ensemble average across 10 folds)
#   3. Train students on c91-labeled pools
# All steps use SLURM --dependency=afterok: for proper chaining.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# Wait-on jobs (c91 10-fold batches)
C91_JOB_IDS="2151798:2151799:2151800:2151801"

C91_POOL_DIR="$REPO/outputs/labeled_pools/k562/ag_s2_c91_10fold"
OUT_BASE="$REPO/outputs/exp1_1/k562_scaling_c91_oracle"
mkdir -p "$C91_POOL_DIR" "$OUT_BASE"

# ── Step 1: Generate c91 pool labels (one SLURM job per reservoir strategy) ──
echo "=== Step 1: Submit c91 pool labeling jobs (depends on c91 10-fold) ==="
STRATEGIES=(random genomic prm_5pct evoaug_structural)
LABEL_JOB_IDS=()
for STRAT in "${STRATEGIES[@]}"; do
    SCRIPT=$(mktemp)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name=c91label_${STRAT}"
        echo "#SBATCH --output=$REPO/logs/%x-%j.out"
        echo "#SBATCH --error=$REPO/logs/%x-%j.err"
        echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=12:00:00 --mem=200G"
        echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
        echo "module load EB5; cd $REPO"
        echo 'export PYTHONPATH="$PWD"'
        echo "source scripts/slurm/setup_hpc_deps.sh"
        echo ""
        echo "# Generate c91-ensemble labels on existing pool for strategy=$STRAT"
        echo "# (Uses generate_stage2_pseudolabels_single_fold.py per fold, then averages.)"
        echo "for FOLD in 0 1 2 3 4 5 6 7 8 9; do"
        echo "  uv run --no-sync python experiments/generate_stage2_pseudolabels_single_fold.py \\"
        echo "    --pool-path $REPO/outputs/labeled_pools/k562/ag_s2/$STRAT/pool.npz \\"
        echo "    --oracle-dir $REPO/outputs/oracle_neg_sweep/debias_c91_10fold/fold_\$FOLD \\"
        echo "    --output-path $C91_POOL_DIR/$STRAT/fold_\$FOLD.npz || true"
        echo "done"
        echo ""
        echo "# Average across folds → final labels"
        echo "uv run --no-sync python -c \"
import numpy as np
from pathlib import Path
strat = '$STRAT'
out_dir = Path('$C91_POOL_DIR') / strat
files = sorted(out_dir.glob('fold_*.npz'))
if not files:
    print(f'No fold files for {strat}'); exit(1)
fold_preds = []
sequences = None
for f in files:
    d = np.load(f, allow_pickle=True)
    fold_preds.append(d['predictions'])
    if sequences is None:
        sequences = d['sequences'] if 'sequences' in d.files else None
mean_pred = np.mean(np.stack(fold_preds), axis=0)
out = out_dir.parent / strat / 'pool.npz'
np.savez_compressed(out, sequences=sequences, labels=mean_pred)
print(f'Saved {out} with {len(mean_pred)} sequences')
\""
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable --dependency=afterok:$C91_JOB_IDS $SCRIPT)
    rm -f $SCRIPT
    echo "  c91label_${STRAT}: $JID (depends on c91 10-fold)"
    LABEL_JOB_IDS+=("$JID")
done

LABEL_JOB_IDS_STR=$(IFS=:; echo "${LABEL_JOB_IDS[*]}")
echo "  All label jobs: $LABEL_JOB_IDS_STR"

# ── Step 2: Submit student training jobs (depend on all label jobs) ──
echo
echo "=== Step 2: Submit student scaling-law jobs (depends on label jobs) ==="
declare -A ARCH_QOS=([legnet]="slow_nice" [dream_rnn]="slow_nice" [dream_attn]="slow_nice")
declare -A ARCH_GPU=([legnet]="v100" [dream_rnn]="v100" [dream_attn]="h100")
declare -A ARCH_K=([legnet]=4 [dream_rnn]=2 [dream_attn]=3)

SMALL_SIZES="1000 5000 10000 30000 100000"
LARGE_SIZES_NONGENOMIC="300000 600000 1000000 2000000"
LARGE_SIZES_GENOMIC="300000 500000"

for ARCH in legnet dream_rnn dream_attn; do
    QOS=${ARCH_QOS[$ARCH]}
    GPU=${ARCH_GPU[$ARCH]}
    for STRAT in "${STRATEGIES[@]}"; do
        if [ "$STRAT" = "genomic" ]; then
            LARGE="$LARGE_SIZES_GENOMIC"
        else
            LARGE="$LARGE_SIZES_NONGENOMIC"
        fi

        SCRIPT=$(mktemp)
        {
            echo "#!/bin/bash"
            echo "#SBATCH --job-name=scl_c91_${ARCH}_${STRAT}"
            echo "#SBATCH --output=$REPO/logs/%x-%j.out"
            echo "#SBATCH --error=$REPO/logs/%x-%j.err"
            echo "#SBATCH --partition=gpuq --qos=$QOS --gres=gpu:${GPU}:1 --cpus-per-task=14 --time=12:00:00 --mem=120G"
            echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
            echo "module load EB5; cd $REPO"
            echo 'export PYTHONPATH="$PWD"'
            echo "export TORCHDYNAMO_DISABLE=1"
            echo "source scripts/slurm/setup_hpc_deps.sh"
            echo ""
            echo "uv run --no-sync python experiments/exp1_1_scaling.py \\"
            echo "    --task k562 --student $ARCH --oracle ag_s2 \\"
            echo "    --reservoir $STRAT \\"
            echo "    --pool-base-dir $C91_POOL_DIR \\"
            echo "    --n-replicates 3 --seed 42 \\"
            echo "    --output-dir $OUT_BASE/${ARCH} \\"
            echo "    --training-sizes $SMALL_SIZES \\"
            echo "    --chr-split --epochs 80 --ensemble-size 1 --early-stop-patience 10 || true"
            echo ""
            echo "uv run --no-sync python experiments/exp1_1_scaling.py \\"
            echo "    --task k562 --student $ARCH --oracle ag_s2 \\"
            echo "    --reservoir $STRAT \\"
            echo "    --pool-base-dir $C91_POOL_DIR \\"
            echo "    --n-replicates 3 --seed 42 \\"
            echo "    --output-dir $OUT_BASE/${ARCH} \\"
            echo "    --training-sizes $LARGE \\"
            echo "    --chr-split --epochs 50 --ensemble-size 1 --early-stop-patience 10 \\"
            echo "    --transfer-hp-from 30000 || true"
        } > $SCRIPT
        JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable --dependency=afterok:$LABEL_JOB_IDS_STR $SCRIPT)
        rm -f $SCRIPT
        echo "  scl_c91_${ARCH}_${STRAT}: $JID (depends on all label jobs)"
    done
done

echo
echo "=== All c91 jobs queued with dependencies ==="
