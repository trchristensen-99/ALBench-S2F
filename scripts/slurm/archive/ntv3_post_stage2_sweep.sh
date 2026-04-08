#!/bin/bash
# NTv3 650M post-trained Stage 2 encoder fine-tuning sweep.
#
# Grid: 3 encoder_lr × 2 unfreeze depths = 6 configs (single seed each).
#   0 → elr=1e-4, last 4 blocks (8-11)
#   1 → elr=1e-4, all 12 blocks (0-11)
#   2 → elr=5e-4, last 4 blocks (8-11)
#   3 → elr=5e-4, all 12 blocks (0-11)
#   4 → elr=1e-3, last 4 blocks (8-11)
#   5 → elr=1e-3, all 12 blocks (0-11)
#
# Prerequisites:
#   NTv3 post-trained grid search must have completed:
#     outputs/foundation_grid_search/ntv3_post/
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ntv3_post_stage2_sweep.sh
#
#SBATCH --job-name=ntv3p_s2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ── Find best Stage 1 config ────────────────────────────────────────────────
S1_BASE="outputs/foundation_grid_search/ntv3_post"
BEST_S1_DIR=$(uv run --no-sync python -c "
import json
from pathlib import Path

base = Path('${S1_BASE}')
best_dir, best_val = None, -1.0
for d in base.iterdir():
    for rfile in d.glob('seed_*/result.json'):
        r = json.load(open(rfile))
        vp = r.get('best_val_pearson_r', 0)
        if vp > best_val:
            best_val = vp
            best_dir = str(rfile.parent)
if best_dir:
    print(best_dir)
else:
    print('NONE')
")

if [ "${BEST_S1_DIR}" = "NONE" ]; then
    echo "ERROR: No Stage 1 results found in ${S1_BASE}"
    exit 1
fi
echo "Best Stage 1 dir: ${BEST_S1_DIR}"

# ── Sweep grid ──────────────────────────────────────────────────────────────
ENCODER_LRS=(1e-4 1e-4 5e-4 5e-4 1e-3 1e-3)
UNFREEZE_SPECS=("8,9,10,11" "0,1,2,3,4,5,6,7,8,9,10,11" \
                "8,9,10,11" "0,1,2,3,4,5,6,7,8,9,10,11" \
                "8,9,10,11" "0,1,2,3,4,5,6,7,8,9,10,11")
LABELS=(uf4 uf12 uf4 uf12 uf4 uf12)

IDX=${SLURM_ARRAY_TASK_ID}
ELR=${ENCODER_LRS[$IDX]}
UF=${UNFREEZE_SPECS[$IDX]}
LBL=${LABELS[$IDX]}

OUT_DIR="outputs/ntv3_post_k562_stage2/sweep_elr${ELR}_${LBL}"

echo "NTv3 post-trained Stage 2 sweep: task=${IDX} encoder_lr=${ELR} unfreeze=${UF}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if result already exists
if [ -f "${OUT_DIR}/result.json" ]; then
    echo "SKIP: result already exists at ${OUT_DIR}/result.json"
    exit 0
fi

uv run --no-sync python experiments/train_ntv3_stage2.py \
    ++stage1_result_dir="${BEST_S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr="${ELR}" \
    ++unfreeze_blocks="${UF}" \
    ++model_variant=post \
    ++seed=42 \
    ++batch_size=16 \
    ++grad_accum_steps=4 \
    ++use_bfloat16=True \
    ++epochs=50 \
    ++early_stop_patience=10

echo "Task ${IDX} DONE — $(date)"

# ── Summary (only on last task) ─────────────────────────────────────────────
if [ "${IDX}" -eq 5 ]; then
    echo ""
    echo "============================================"
    echo "=== NTv3 POST-TRAINED Stage 2 SWEEP SUMMARY ==="
    echo "============================================"
    uv run --no-sync python -c "
import json
from pathlib import Path

base = Path('outputs/ntv3_post_k562_stage2')
results = []
for d in sorted(base.glob('sweep_*')):
    rfile = d / 'result.json'
    if not rfile.exists():
        continue
    r = json.load(open(rfile))
    tm = r.get('test_metrics', {})
    results.append({
        'config': d.name,
        'val_pearson': r.get('best_val_pearson', 0),
        'in_dist': tm.get('in_distribution', {}).get('pearson_r', 0),
        'snv_abs': tm.get('snv_abs', {}).get('pearson_r', 0),
        'ood': tm.get('ood', {}).get('pearson_r', 0),
    })

results.sort(key=lambda x: x['in_dist'], reverse=True)
print(f\"{'Config':<35} {'Val':>8} {'InDist':>8} {'SNV':>8} {'OOD':>8}\")
print('-' * 75)
for r in results:
    print(f\"{r['config']:<35} {r['val_pearson']:>8.4f} {r['in_dist']:>8.4f} {r['snv_abs']:>8.4f} {r['ood']:>8.4f}\")
" || echo "Summary generation failed"
fi
