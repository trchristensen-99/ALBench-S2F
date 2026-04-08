#!/bin/bash
# Foundation model head hyperparameter grid search.
# Sweeps lr × weight_decay × dropout on cached embeddings (27 configs per model).
# Each config trains in ~5-10 min → ~3-4h per model on a single GPU.
#
# Array: 0=NTv3, 1=Borzoi, 2=Enformer
#
# Submit NTv3 + Borzoi (caches ready):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-1 scripts/slurm/foundation_grid_search.sh
#
# Submit Enformer (after cache job completes):
#   /cm/shared/apps/slurm/current/bin/sbatch --array=2 --dependency=afterok:<CACHE_JOBID> scripts/slurm/foundation_grid_search.sh
#
#SBATCH --job-name=fm_grid
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# Model configs
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL=ntv3
    CACHE_DIR=outputs/ntv3_k562_cached/embedding_cache
    EMBED_DIM=1536
    ;;
  1)
    MODEL=borzoi
    CACHE_DIR=outputs/borzoi_k562_cached/embedding_cache
    EMBED_DIM=1536
    ;;
  2)
    MODEL=enformer
    CACHE_DIR=outputs/enformer_k562_cached/embedding_cache
    EMBED_DIM=3072
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

OUT_BASE="outputs/foundation_grid_search/${MODEL}"

echo "Foundation model grid search: ${MODEL} (task ${SLURM_ARRAY_TASK_ID})"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Grid
LRS=(0.0001 0.0005 0.001)
WDS=(0.000001 0.0001 0.001)
DROPOUTS=(0.1 0.3 0.5)

TOTAL=$((${#LRS[@]} * ${#WDS[@]} * ${#DROPOUTS[@]}))
COUNT=0

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    for do_val in "${DROPOUTS[@]}"; do
      COUNT=$((COUNT + 1))
      OUT_DIR="${OUT_BASE}/lr${lr}_wd${wd}_do${do_val}"

      # Skip if result already exists
      if ls "${OUT_DIR}"/seed_*/result.json > /dev/null 2>&1; then
        echo "[${COUNT}/${TOTAL}] SKIP: lr=${lr} wd=${wd} do=${do_val} (already done)"
        continue
      fi

      echo ""
      echo "[${COUNT}/${TOTAL}] ${MODEL}: lr=${lr} wd=${wd} dropout=${do_val}"
      echo "Start: $(date)"

      uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name="${MODEL}" \
        ++cache_dir="${CACHE_DIR}" \
        ++embed_dim="${EMBED_DIM}" \
        ++output_dir="${OUT_DIR}" \
        ++lr="${lr}" \
        ++weight_decay="${wd}" \
        ++dropout="${do_val}" \
        ++seed=42 \
        || echo "FAILED: lr=${lr} wd=${wd} dropout=${do_val}"

      echo "Done: $(date)"
    done
  done
done

# Print summary table
echo ""
echo "============================================"
echo "=== ${MODEL} GRID SEARCH SUMMARY ==="
echo "============================================"
echo ""

uv run --no-sync python -c "
import json
from pathlib import Path

base = Path('${OUT_BASE}')
results = []
for d in sorted(base.iterdir()):
    rfiles = list(d.glob('seed_*/result.json'))
    if not rfiles:
        continue
    r = json.load(open(rfiles[0]))
    tm = r.get('test_metrics', {})
    results.append({
        'config': d.name,
        'val_pearson': r.get('best_val_pearson_r', 0),
        'in_dist': tm.get('in_distribution', {}).get('pearson_r', 0),
        'snv_abs': tm.get('snv_abs', {}).get('pearson_r', 0),
        'ood': tm.get('ood', {}).get('pearson_r', 0),
    })

results.sort(key=lambda x: x['in_dist'], reverse=True)
print(f\"{'Config':<35} {'Val':>8} {'InDist':>8} {'SNV':>8} {'OOD':>8}\")
print('-' * 75)
for r in results:
    print(f\"{r['config']:<35} {r['val_pearson']:>8.4f} {r['in_dist']:>8.4f} {r['snv_abs']:>8.4f} {r['ood']:>8.4f}\")

print()
best = results[0]
print(f\"Best config: {best['config']}\")
print(f\"  in_dist Pearson: {best['in_dist']:.4f}\")
print(f\"  SNV abs Pearson: {best['snv_abs']:.4f}\")
print(f\"  OOD Pearson:     {best['ood']:.4f}\")
" || echo "Summary generation failed"

echo ""
echo "=== ${MODEL} grid search COMPLETE — $(date) ==="
