#!/bin/bash
# Borzoi Stage 2 encoder fine-tuning sweep on K562 hashFrag.
#
# Grid: 3 encoder_lr x transformer-only = 3 configs (single seed each).
# "all" unfreeze mode is unstable for Borzoi (zero-padded conv gradients).
#   0 -> elr=1e-6, unfreeze=transformer
#   1 -> elr=1e-5, unfreeze=transformer
#   2 -> elr=1e-4, unfreeze=transformer
#
# Prerequisites:
#   Stage 1 grid search must have at least one completed result in:
#     outputs/foundation_grid_search/borzoi/
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/borzoi_stage2_sweep.sh
#
#SBATCH --job-name=borzoi_s2_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ── Find best Stage 1 config ────────────────────────────────────────────────
S1_BASE="outputs/foundation_grid_search/borzoi"
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
ENCODER_LRS=(1e-6 1e-5 1e-4)
UNFREEZE_MODES=(transformer transformer transformer)
LABELS=(elr1e-6_transformer elr1e-5_transformer elr1e-4_transformer)

IDX=${SLURM_ARRAY_TASK_ID}
ELR=${ENCODER_LRS[$IDX]}
UFM=${UNFREEZE_MODES[$IDX]}
LBL=${LABELS[$IDX]}

OUT_DIR="outputs/borzoi_k562_stage2/sweep_${LBL}"

echo "Borzoi Stage 2 sweep: task=${IDX} encoder_lr=${ELR} unfreeze=${UFM}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if result already exists
if [ -f "${OUT_DIR}/result.json" ]; then
    echo "SKIP: result already exists at ${OUT_DIR}/result.json"
    exit 0
fi

uv run --no-sync python experiments/train_foundation_stage2.py \
    ++model_name=borzoi \
    ++stage1_result_dir="${BEST_S1_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++encoder_lr="${ELR}" \
    ++unfreeze_mode="${UFM}" \
    ++seed=42 \
    ++batch_size=4 \
    ++grad_accum_steps=2 \
    ++epochs=15 \
    ++early_stop_patience=5 \
    ++max_train_sequences=20000 \
    ++max_val_sequences=2000 \
    ++use_amp=false

echo "Task ${IDX} DONE — $(date)"

# ── Summary (only on last task) ─────────────────────────────────────────────
if [ "${IDX}" -eq 2 ]; then
    echo ""
    echo "============================================"
    echo "=== Borzoi Stage 2 SWEEP SUMMARY ==="
    echo "============================================"
    uv run --no-sync python -c "
import json
from pathlib import Path

base = Path('outputs/borzoi_k562_stage2')
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
