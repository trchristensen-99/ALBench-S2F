#!/bin/bash
# Auto-detect best grid search configs and submit final 3-seed evaluations.
# Also submits NTv3 Stage 2 sweep.
#
# Run this AFTER grid searches complete:
#   bash scripts/slurm/submit_finals_auto.sh
#
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

# ── Helper: find best config from grid search results ─────────────────────
best_config() {
    local base_dir="$1"
    uv run --no-sync python -c "
import json
from pathlib import Path

base = Path('${base_dir}')
best_dir, best_val, best_cfg = None, -1.0, ''
for d in base.iterdir():
    for rfile in d.glob('seed_*/result.json'):
        r = json.load(open(rfile))
        vp = r.get('best_val_pearson_r', 0)
        if vp > best_val:
            best_val = vp
            best_dir = str(rfile.parent)
            best_cfg = d.name
if best_cfg:
    # Parse lr, wd, do from dir name like 'lr0.0001_wd0.000001_do0.1'
    parts = best_cfg.split('_')
    lr = parts[0].replace('lr', '')
    wd = parts[1].replace('wd', '')
    do = parts[2].replace('do', '')
    print(f'{lr} {wd} {do} {best_val:.4f} {best_dir}')
else:
    print('NONE')
"
}

echo "=========================================="
echo "  Auto-submit final 3-seed evaluations"
echo "=========================================="
echo ""

# ── NTv3 ──────────────────────────────────────────────────────────────────
echo "=== NTv3 ==="
NTV3_RESULT=$(best_config "outputs/foundation_grid_search/ntv3")
if [ "$NTV3_RESULT" = "NONE" ]; then
    echo "  ERROR: No NTv3 grid search results found"
else
    NTV3_LR=$(echo "$NTV3_RESULT" | awk '{print $1}')
    NTV3_WD=$(echo "$NTV3_RESULT" | awk '{print $2}')
    NTV3_DO=$(echo "$NTV3_RESULT" | awk '{print $3}')
    NTV3_VAL=$(echo "$NTV3_RESULT" | awk '{print $4}')
    NTV3_DIR=$(echo "$NTV3_RESULT" | awk '{print $5}')
    echo "  Best: lr=${NTV3_LR} wd=${NTV3_WD} do=${NTV3_DO} (val=${NTV3_VAL})"

    # Submit 3-seed S1 final
    for SEED_IDX in 0 1 2; do
        JOB=$($SBATCH --parsable \
            --job-name=ntv3_s1_final \
            --output=logs/%x-%j.out \
            --error=logs/%x-%j.err \
            --partition=gpuq --qos=slow_nice \
            --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
            --wrap="
set +u; source /etc/profile.d/modules.sh; set -u; module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python experiments/train_foundation_cached.py \
    ++model_name=ntv3 \
    ++cache_dir=outputs/ntv3_k562_cached/embedding_cache \
    ++embed_dim=1536 \
    ++output_dir=outputs/ntv3_k562_3seeds \
    ++lr=${NTV3_LR} ++weight_decay=${NTV3_WD} ++dropout=${NTV3_DO}
")
        echo "  NTv3 S1 seed ${SEED_IDX}: job ${JOB}"
    done

    # Submit NTv3 Stage 2 sweep
    S2_JOB=$($SBATCH --parsable scripts/slurm/ntv3_stage2_sweep.sh)
    echo "  NTv3 S2 sweep: job ${S2_JOB} (array 0-5)"
fi

echo ""

# ── Borzoi ────────────────────────────────────────────────────────────────
echo "=== Borzoi ==="
BORZOI_RESULT=$(best_config "outputs/foundation_grid_search/borzoi")
if [ "$BORZOI_RESULT" = "NONE" ]; then
    echo "  ERROR: No Borzoi grid search results found"
else
    BORZOI_LR=$(echo "$BORZOI_RESULT" | awk '{print $1}')
    BORZOI_WD=$(echo "$BORZOI_RESULT" | awk '{print $2}')
    BORZOI_DO=$(echo "$BORZOI_RESULT" | awk '{print $3}')
    BORZOI_VAL=$(echo "$BORZOI_RESULT" | awk '{print $4}')
    echo "  Best: lr=${BORZOI_LR} wd=${BORZOI_WD} do=${BORZOI_DO} (val=${BORZOI_VAL})"

    for SEED_IDX in 0 1 2; do
        JOB=$($SBATCH --parsable \
            --job-name=borzoi_s1_final \
            --output=logs/%x-%j.out \
            --error=logs/%x-%j.err \
            --partition=gpuq --qos=slow_nice \
            --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
            --wrap="
set +u; source /etc/profile.d/modules.sh; set -u; module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python experiments/train_foundation_cached.py \
    ++model_name=borzoi \
    ++cache_dir=outputs/borzoi_k562_cached/embedding_cache \
    ++embed_dim=1536 \
    ++output_dir=outputs/borzoi_k562_3seeds \
    ++lr=${BORZOI_LR} ++weight_decay=${BORZOI_WD} ++dropout=${BORZOI_DO}
")
        echo "  Borzoi S1 seed ${SEED_IDX}: job ${JOB}"
    done
fi

echo ""

# ── Enformer ──────────────────────────────────────────────────────────────
echo "=== Enformer ==="
ENFORMER_RESULT=$(best_config "outputs/foundation_grid_search/enformer")
if [ "$ENFORMER_RESULT" = "NONE" ]; then
    echo "  SKIP: No Enformer grid search results yet (cache may still be building)"
else
    ENFORMER_LR=$(echo "$ENFORMER_RESULT" | awk '{print $1}')
    ENFORMER_WD=$(echo "$ENFORMER_RESULT" | awk '{print $2}')
    ENFORMER_DO=$(echo "$ENFORMER_RESULT" | awk '{print $3}')
    ENFORMER_VAL=$(echo "$ENFORMER_RESULT" | awk '{print $4}')
    echo "  Best: lr=${ENFORMER_LR} wd=${ENFORMER_WD} do=${ENFORMER_DO} (val=${ENFORMER_VAL})"

    for SEED_IDX in 0 1 2; do
        JOB=$($SBATCH --parsable \
            --job-name=enformer_s1_final \
            --output=logs/%x-%j.out \
            --error=logs/%x-%j.err \
            --partition=gpuq --qos=slow_nice \
            --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
            --wrap="
set +u; source /etc/profile.d/modules.sh; set -u; module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python experiments/train_foundation_cached.py \
    ++model_name=enformer \
    ++cache_dir=outputs/enformer_k562_cached/embedding_cache \
    ++embed_dim=3072 \
    ++output_dir=outputs/enformer_k562_3seeds \
    ++lr=${ENFORMER_LR} ++weight_decay=${ENFORMER_WD} ++dropout=${ENFORMER_DO}
")
        echo "  Enformer S1 seed ${SEED_IDX}: job ${JOB}"
    done
fi

echo ""
echo "Done. Check jobs with: /cm/shared/apps/slurm/current/bin/squeue -u christen"
