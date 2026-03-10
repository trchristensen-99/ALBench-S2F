#!/bin/bash
# AlphaGenome yeast head hyperparameter grid search.
# Sweeps lr × weight_decay × dropout on cached embeddings (27 configs).
# Uses f=1.0, seed=42, frozen encoder (cached).
# Each config trains in ~10-20 min → ~5-9h total on a single GPU.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ag_yeast_head_grid_search.sh
#
#SBATCH --job-name=ag_yeast_grid
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

OUT_BASE="outputs/foundation_grid_search/alphagenome"

echo "AG yeast head hyperparameter grid search"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

# Grid: 3 lr × 3 wd × 3 dropout = 27 configs
LRS=(0.0001 0.0005 0.001)
WDS=(0.000001 0.0001 0.001)
DROPOUTS=(0.1 0.3 0.5)

TOTAL=$((${#LRS[@]} * ${#WDS[@]} * ${#DROPOUTS[@]}))
COUNT=0

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for do_val in "${DROPOUTS[@]}"; do
      COUNT=$((COUNT + 1))
      TAG="lr${lr}_wd${wd}_do${do_val}"
      OUT_DIR="${OUT_BASE}/${TAG}"

      # Skip if result already exists
      if ls "${OUT_DIR}"/fraction_*/seed_*/result.json > /dev/null 2>&1; then
        echo "[${COUNT}/${TOTAL}] SKIP: ${TAG} (already done)"
        continue
      fi

      echo ""
      echo "[${COUNT}/${TOTAL}] AG yeast: lr=${lr} wd=${wd} dropout=${do_val}"
      echo "Start: $(date)"

      uv run --no-sync python experiments/exp0_yeast_scaling_alphagenome.py \
        ++fraction=1.0 \
        ++seed=42 \
        ++output_dir="${OUT_DIR}" \
        ++lr="${lr}" \
        ++weight_decay="${wd}" \
        ++dropout_rate="${do_val}" \
        ++epochs=50 \
        ++early_stop_patience=7 \
        ++wandb_mode=offline \
        ++test_subset_dir=data/yeast/test_subset_ids \
        || echo "FAILED: ${TAG}"

      echo "Done: $(date)"
    done
  done
done

# Print summary table
echo ""
echo "============================================"
echo "=== AG YEAST GRID SEARCH SUMMARY ==="
echo "============================================"
echo ""

uv run --no-sync python -c "
import json
from pathlib import Path

base = Path('${OUT_BASE}')
results = []
for d in sorted(base.iterdir()):
    rfiles = list(d.glob('fraction_*/seed_*/result.json'))
    if not rfiles:
        continue
    r = json.load(open(rfiles[0]))
    tm = r.get('test_metrics', {})
    results.append({
        'config': d.name,
        'val_pearson': r.get('best_val_pearson_r', 0),
        'random': tm.get('random', {}).get('pearson_r', 0),
        'snv_abs': tm.get('snv_abs', {}).get('pearson_r', 0),
        'genomic': tm.get('genomic', {}).get('pearson_r', 0),
    })

results.sort(key=lambda x: x['random'], reverse=True)
print(f\"{'Config':<35} {'Val':>8} {'Random':>8} {'SNV':>8} {'Genomic':>8}\")
print('-' * 75)
for r in results:
    print(f\"{r['config']:<35} {r['val_pearson']:>8.4f} {r['random']:>8.4f} {r['snv_abs']:>8.4f} {r['genomic']:>8.4f}\")

print()
best = results[0]
print(f\"Best config: {best['config']}\")
print(f\"  Random Pearson:  {best['random']:.4f}\")
print(f\"  SNV abs Pearson: {best['snv_abs']:.4f}\")
print(f\"  Genomic Pearson: {best['genomic']:.4f}\")
" || echo "Summary generation failed"

echo ""
echo "=== AG yeast grid search COMPLETE — $(date) ==="
