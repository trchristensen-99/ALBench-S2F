#!/bin/bash
# Submit the complete overnight pipeline for foundation model final results.
#
# Chain:
#   1. Wait for grid searches (NTv3 830075, Borzoi 830076) to finish
#   2. Submit S1 3-seed finals for NTv3, Borzoi (auto-detect best config)
#   3. Wait for Enformer cache (830073) + grid search (830074) to finish
#   4. Submit Enformer S1 3-seed final
#   5. NTv3 S2 sweep (830084) already running → submit S2 3-seed final after
#
# Usage:
#   bash scripts/slurm/submit_overnight_pipeline.sh
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
    parts = best_cfg.split('_')
    lr = parts[0].replace('lr', '')
    wd = parts[1].replace('wd', '')
    do = parts[2].replace('do', '')
    print(f'{lr} {wd} {do} {best_val:.4f} {best_dir}')
else:
    print('NONE')
"
}

# ── Helper: submit 3 seeds for a model ────────────────────────────────────
submit_3seeds() {
    local MODEL="$1" CACHE_DIR="$2" EMBED_DIM="$3" OUT_DIR="$4" LR="$5" WD="$6" DO="$7"
    local DEP_ARG="${8:-}"

    for SEED_IDX in 0 1 2; do
        local JOB
        JOB=$($SBATCH --parsable \
            ${DEP_ARG:+--dependency=afterok:${DEP_ARG}} \
            --job-name="${MODEL}_final" \
            --output="logs/%x-%j.out" \
            --error="logs/%x-%j.err" \
            --partition=gpuq --qos=slow_nice \
            --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \
            --wrap="
set +u; source /etc/profile.d/modules.sh; set -u; module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python experiments/train_foundation_cached.py \
    ++model_name=${MODEL} \
    ++cache_dir=${CACHE_DIR} \
    ++embed_dim=${EMBED_DIM} \
    ++output_dir=${OUT_DIR} \
    ++lr=${LR} ++weight_decay=${WD} ++dropout=${DO}
echo '${MODEL} seed ${SEED_IDX} DONE'
")
        echo "  ${MODEL} final seed ${SEED_IDX}: job ${JOB}"
    done
}

echo "=========================================="
echo "  Overnight Pipeline Submission"
echo "  $(date)"
echo "=========================================="
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# 1. NTv3 S1 3-seed final (depends on grid search 830075)
# ═══════════════════════════════════════════════════════════════════════════
echo "=== NTv3 S1 ==="
NTV3_RESULT=$(best_config "outputs/foundation_grid_search/ntv3")
if [ "$NTV3_RESULT" = "NONE" ]; then
    echo "  ERROR: No NTv3 grid search results"
else
    NTV3_LR=$(echo "$NTV3_RESULT" | awk '{print $1}')
    NTV3_WD=$(echo "$NTV3_RESULT" | awk '{print $2}')
    NTV3_DO=$(echo "$NTV3_RESULT" | awk '{print $3}')
    NTV3_VAL=$(echo "$NTV3_RESULT" | awk '{print $4}')
    echo "  Best config: lr=${NTV3_LR} wd=${NTV3_WD} do=${NTV3_DO} (val=${NTV3_VAL})"
    submit_3seeds ntv3 \
        outputs/ntv3_k562_cached/embedding_cache 1536 \
        outputs/ntv3_k562_3seeds \
        "$NTV3_LR" "$NTV3_WD" "$NTV3_DO" \
        830075
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# 2. Borzoi S1 3-seed final (depends on grid search 830076)
# ═══════════════════════════════════════════════════════════════════════════
echo "=== Borzoi S1 ==="
BORZOI_RESULT=$(best_config "outputs/foundation_grid_search/borzoi")
if [ "$BORZOI_RESULT" = "NONE" ]; then
    echo "  ERROR: No Borzoi grid search results"
else
    BORZOI_LR=$(echo "$BORZOI_RESULT" | awk '{print $1}')
    BORZOI_WD=$(echo "$BORZOI_RESULT" | awk '{print $2}')
    BORZOI_DO=$(echo "$BORZOI_RESULT" | awk '{print $3}')
    BORZOI_VAL=$(echo "$BORZOI_RESULT" | awk '{print $4}')
    echo "  Best config: lr=${BORZOI_LR} wd=${BORZOI_WD} do=${BORZOI_DO} (val=${BORZOI_VAL})"
    submit_3seeds borzoi \
        outputs/borzoi_k562_cached/embedding_cache 1536 \
        outputs/borzoi_k562_3seeds \
        "$BORZOI_LR" "$BORZOI_WD" "$BORZOI_DO" \
        830076
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# 3. Enformer S1 3-seed final (depends on grid search 830074)
# ═══════════════════════════════════════════════════════════════════════════
echo "=== Enformer S1 ==="
# Enformer grid hasn't run yet, so we can't know best config.
# Submit a wrapper job that runs after grid search, finds best, and trains 3 seeds.
ENFORMER_META=$($SBATCH --parsable \
    --dependency=afterok:830074 \
    --job-name=enformer_final_meta \
    --output=logs/%x-%j.out \
    --error=logs/%x-%j.err \
    --partition=gpuq --qos=slow_nice \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
    --wrap="
set +u; source /etc/profile.d/modules.sh; set -u; module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"
source scripts/slurm/setup_hpc_deps.sh

# Find best Enformer config
BEST=\$(uv run --no-sync python -c \"
import json
from pathlib import Path
base = Path('outputs/foundation_grid_search/enformer')
best_val, best_cfg = -1.0, ''
for d in base.iterdir():
    for rfile in d.glob('seed_*/result.json'):
        r = json.load(open(rfile))
        vp = r.get('best_val_pearson_r', 0)
        if vp > best_val:
            best_val = vp
            best_cfg = d.name
if best_cfg:
    parts = best_cfg.split('_')
    lr = parts[0].replace('lr', '')
    wd = parts[1].replace('wd', '')
    do_val = parts[2].replace('do', '')
    print(f'{lr} {wd} {do_val} {best_val:.4f}')
else:
    print('NONE')
\")

if [ \"\$BEST\" = 'NONE' ]; then
    echo 'ERROR: No Enformer grid results'
    exit 1
fi

LR=\$(echo \"\$BEST\" | awk '{print \$1}')
WD=\$(echo \"\$BEST\" | awk '{print \$2}')
DO=\$(echo \"\$BEST\" | awk '{print \$3}')
echo \"Best Enformer config: lr=\${LR} wd=\${WD} do=\${DO}\"

# Train 3 seeds sequentially (all on this GPU)
for SEED_IDX in 0 1 2; do
    echo \"Training seed \${SEED_IDX}...\"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name=enformer \
        ++cache_dir=outputs/enformer_k562_cached/embedding_cache \
        ++embed_dim=3072 \
        ++output_dir=outputs/enformer_k562_3seeds \
        ++lr=\${LR} ++weight_decay=\${WD} ++dropout=\${DO}
    echo \"Seed \${SEED_IDX} DONE\"
done
echo 'All 3 Enformer seeds DONE'
")
echo "  Enformer meta job (finds best + trains 3 seeds): ${ENFORMER_META}"
echo "  Depends on: Enformer grid search (830074) → cache (830073)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# 4. NTv3 S2 3-seed final (depends on S2 sweep 830084)
# ═══════════════════════════════════════════════════════════════════════════
echo "=== NTv3 S2 ==="
# Submit a meta job that finds best S2 config and trains 3 seeds
NTV3_S2_META=$($SBATCH --parsable \
    --dependency=afterok:830084 \
    --job-name=ntv3_s2_final_meta \
    --output=logs/%x-%j.out \
    --error=logs/%x-%j.err \
    --partition=gpuq --qos=slow_nice \
    --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=48:00:00 \
    --wrap="
set +u; source /etc/profile.d/modules.sh; set -u; module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"
source scripts/slurm/setup_hpc_deps.sh

# Find best S2 sweep config
BEST_S2=\$(uv run --no-sync python -c \"
import json
from pathlib import Path
base = Path('outputs/ntv3_k562_stage2')
best_val, best_elr, best_uf = -1.0, '', ''
for d in sorted(base.glob('sweep_*')):
    rfile = d / 'result.json'
    if not rfile.exists():
        continue
    r = json.load(open(rfile))
    vp = r.get('best_val_pearson', 0)
    cfg = r.get('config', {})
    if vp > best_val:
        best_val = vp
        best_elr = str(cfg.get('encoder_lr', ''))
        best_uf = str(cfg.get('unfreeze_blocks', ''))
if best_elr:
    print(f'{best_elr} {best_uf} {best_val:.4f}')
else:
    print('NONE')
\")

if [ \"\$BEST_S2\" = 'NONE' ]; then
    echo 'ERROR: No NTv3 S2 sweep results'
    exit 1
fi

BEST_ELR=\$(echo \"\$BEST_S2\" | awk '{print \$1}')
BEST_UF=\$(echo \"\$BEST_S2\" | awk '{print \$2}')
echo \"Best S2 config: encoder_lr=\${BEST_ELR} unfreeze=\${BEST_UF}\"

# Find best S1 checkpoint
S1_DIR=\$(uv run --no-sync python -c \"
import json
from pathlib import Path
base = Path('outputs/foundation_grid_search/ntv3')
best_dir, best_val = None, -1.0
for d in base.iterdir():
    for rfile in d.glob('seed_*/result.json'):
        r = json.load(open(rfile))
        vp = r.get('best_val_pearson_r', 0)
        if vp > best_val:
            best_val = vp
            best_dir = str(rfile.parent)
print(best_dir if best_dir else 'NONE')
\")

echo \"Using S1 checkpoint: \${S1_DIR}\"

# Train 3 seeds sequentially
for SEED_IDX in 0 1 2; do
    echo \"\"
    echo \"=== NTv3 S2 final seed \${SEED_IDX} ===\"
    uv run --no-sync python experiments/train_ntv3_stage2.py \
        ++stage1_result_dir=\"\${S1_DIR}\" \
        ++output_dir=\"outputs/ntv3_k562_stage2_final/run_\${SEED_IDX}\" \
        ++encoder_lr=\"\${BEST_ELR}\" \
        ++unfreeze_blocks=\"\${BEST_UF}\" \
        ++batch_size=64 \
        ++epochs=50 \
        ++early_stop_patience=10
    echo \"Seed \${SEED_IDX} DONE — \$(date)\"
done
echo 'All 3 NTv3 S2 seeds DONE'
")
echo "  NTv3 S2 meta job (finds best + trains 3 seeds): ${NTV3_S2_META}"
echo "  Depends on: S2 sweep (830084)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
echo "=========================================="
echo "  All jobs submitted. Pipeline:"
echo "  NTv3 grid (830075) → NTv3 S1 3-seed"
echo "  Borzoi grid (830076) → Borzoi S1 3-seed"
echo "  Enformer cache (830073) → grid (830074) → Enformer 3-seed"
echo "  NTv3 S2 sweep (830084) → NTv3 S2 3-seed"
echo "=========================================="
echo ""
echo "Monitor: /cm/shared/apps/slurm/current/bin/squeue -u christen"
