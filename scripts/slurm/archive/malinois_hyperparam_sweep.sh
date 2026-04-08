#!/bin/bash
# Malinois hyperparameter sweep on K562 hashFrag with 3 seeds each.
#
# Grid: 4 configs × 3 seeds = 12 tasks.
# Current baseline: lr=0.00327, wd=3.44e-4, bs=512
# Sweep: LR (higher/lower) × WD (higher/lower)
#
#   Config 0: lr=0.001,  wd=1e-4   (lower LR, lower WD)
#   Config 1: lr=0.001,  wd=1e-3   (lower LR, higher WD)
#   Config 2: lr=0.005,  wd=1e-4   (higher LR, lower WD)
#   Config 3: lr=0.005,  wd=1e-3   (higher LR, higher WD)
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/malinois_hyperparam_sweep.sh
#
#SBATCH --job-name=malinois_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-11

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# ── Sweep grid: 4 configs × 3 seeds ─────────────────────────────────────────
LRS=(0.001 0.001 0.005 0.005)
WDS=(1e-4 1e-3 1e-4 1e-3)
LABELS=(lr0.001_wd1e-4 lr0.001_wd1e-3 lr0.005_wd1e-4 lr0.005_wd1e-3)

IDX=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$((IDX / 3))
SEED_IDX=$((IDX % 3))

LR=${LRS[$CONFIG_IDX]}
WD=${WDS[$CONFIG_IDX]}
LBL=${LABELS[$CONFIG_IDX]}

OUT_DIR="outputs/malinois_k562_sweep/${LBL}/seed_${SEED_IDX}"

echo "Malinois sweep: task=${IDX} config=${LBL} seed_idx=${SEED_IDX}"
echo "  lr=${LR} wd=${WD}"
echo "Output: ${OUT_DIR}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Skip if result already exists
if [ -f "${OUT_DIR}/result.json" ]; then
    echo "SKIP: result already exists at ${OUT_DIR}/result.json"
    exit 0
fi

uv run --no-sync python experiments/train_malinois_k562.py \
    ++output_dir="${OUT_DIR}" \
    ++lr="${LR}" \
    ++weight_decay="${WD}" \
    ++epochs=200 \
    ++early_stop_patience=15 \
    ++batch_size=512

echo "Task ${IDX} DONE — $(date)"

# ── Summary (only on last task) ─────────────────────────────────────────────
if [ "${IDX}" -eq 11 ]; then
    echo ""
    echo "============================================"
    echo "=== Malinois SWEEP SUMMARY ==="
    echo "============================================"
    uv run --no-sync python -c "
import json
from pathlib import Path
import numpy as np

base = Path('outputs/malinois_k562_sweep')
configs = {}
for d in sorted(base.glob('*/seed_*')):
    rfile = d / 'result.json'
    if not rfile.exists():
        continue
    r = json.load(open(rfile))
    tm = r.get('test_metrics', {})
    config = d.parent.name
    if config not in configs:
        configs[config] = []
    configs[config].append({
        'in_dist': tm.get('in_distribution', {}).get('pearson_r', 0),
        'snv_abs': tm.get('snv_abs', {}).get('pearson_r', 0),
        'ood': tm.get('ood', {}).get('pearson_r', 0),
    })

print(f\"{'Config':<25} {'InDist':>12} {'SNV':>12} {'OOD':>12} {'N':>4}\")
print('-' * 70)
for config in sorted(configs.keys()):
    runs = configs[config]
    ids = [r['in_dist'] for r in runs]
    snvs = [r['snv_abs'] for r in runs]
    oods = [r['ood'] for r in runs]
    n = len(runs)
    print(f\"{config:<25} {np.mean(ids):>5.4f}±{np.std(ids):>5.4f} {np.mean(snvs):>5.4f}±{np.std(snvs):>5.4f} {np.mean(oods):>5.4f}±{np.std(oods):>5.4f} {n:>4}\")
" || echo "Summary generation failed"
fi
