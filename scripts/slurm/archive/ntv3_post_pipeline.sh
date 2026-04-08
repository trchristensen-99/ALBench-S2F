#!/bin/bash
# NTv3 650M post-trained (species-conditioned) full pipeline.
# Phase 1: Build embedding cache (~3-4h)
# Phase 2: 6-config sweep × 3 seeds = 18 runs (~2-3h)
#   Focused grid: lr ∈ {0.0005, 0.001, 0.002} × wd ∈ {1e-6, 1e-4} × dropout=0.1
#   3 seeds per config → results are final (mean ± std).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/ntv3_post_pipeline.sh
#
#SBATCH --job-name=ntv3_post
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

CACHE_DIR="outputs/ntv3_post_k562_cached/embedding_cache"
GRID_BASE="outputs/foundation_grid_search/ntv3_post"

# ── Phase 1: Build embedding cache ──────────────────────────────────────────
echo "=== Phase 1: Build NTv3 650M post-trained cache (1536D) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python scripts/build_ntv3_embedding_cache.py \
    --data-path data/k562 \
    --cache-dir "${CACHE_DIR}" \
    --splits train val \
    --include-test \
    --batch-size 32 \
    --model-variant post

echo "Cache build DONE — $(date)"

# ── Phase 2: Focused sweep — 6 configs × 3 seeds ───────────────────────────
echo ""
echo "=== Phase 2: Focused sweep (6 configs × 3 seeds = 18 runs) ==="
echo "Start: $(date)"

LRS=(0.0005 0.001 0.002)
WDS=(0.000001 0.0001)
DROPOUT=0.1
SEEDS=(42 123 456)

TOTAL=$((${#LRS[@]} * ${#WDS[@]} * ${#SEEDS[@]}))
COUNT=0

for lr in "${LRS[@]}"; do
  for wd in "${WDS[@]}"; do
    OUT_DIR="${GRID_BASE}/lr${lr}_wd${wd}_do${DROPOUT}"

    for seed in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))

      # Skip if this seed already done
      if [ -f "${OUT_DIR}/seed_${seed}/result.json" ]; then
        echo "[${COUNT}/${TOTAL}] SKIP: lr=${lr} wd=${wd} seed=${seed} (already done)"
        continue
      fi

      echo ""
      echo "[${COUNT}/${TOTAL}] ntv3_post: lr=${lr} wd=${wd} do=${DROPOUT} seed=${seed}"
      echo "Start: $(date)"

      uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name=ntv3_post \
        ++cache_dir="${CACHE_DIR}" \
        ++embed_dim=1536 \
        ++output_dir="${OUT_DIR}" \
        ++lr="${lr}" \
        ++weight_decay="${wd}" \
        ++dropout="${DROPOUT}" \
        ++seed="${seed}" \
        || echo "FAILED: lr=${lr} wd=${wd} seed=${seed}"

      echo "Done: $(date)"
    done
  done
done

echo ""
echo "=== Sweep DONE — $(date) ==="

# ── Print summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "=== NTv3 POST-TRAINED SWEEP SUMMARY ==="
echo "============================================"

uv run --no-sync python -c "
import json, numpy as np
from pathlib import Path

base = Path('${GRID_BASE}')
configs = {}
for d in sorted(base.iterdir()):
    rfiles = sorted(d.glob('seed_*/result.json'))
    if not rfiles:
        continue
    vals, ids, snvs, oods = [], [], [], []
    for rfile in rfiles:
        r = json.load(open(rfile))
        tm = r.get('test_metrics', {})
        vals.append(r.get('best_val_pearson_r', 0))
        ids.append(tm.get('in_distribution', {}).get('pearson_r', 0))
        snvs.append(tm.get('snv_abs', {}).get('pearson_r', 0))
        oods.append(tm.get('ood', {}).get('pearson_r', 0))
    configs[d.name] = {
        'n_seeds': len(rfiles),
        'val': (np.mean(vals), np.std(vals)),
        'in_dist': (np.mean(ids), np.std(ids)),
        'snv_abs': (np.mean(snvs), np.std(snvs)),
        'ood': (np.mean(oods), np.std(oods)),
    }

# Sort by mean in_dist Pearson
ranked = sorted(configs.items(), key=lambda x: x[1]['in_dist'][0], reverse=True)
print(f\"{'Config':<35} {'N':>3} {'Val':>14} {'InDist':>14} {'SNV':>14} {'OOD':>14}\")
print('-' * 100)
for name, m in ranked:
    print(f\"{name:<35} {m['n_seeds']:>3} \"
          f\"{m['val'][0]:>6.4f}+/-{m['val'][1]:.4f} \"
          f\"{m['in_dist'][0]:>6.4f}+/-{m['in_dist'][1]:.4f} \"
          f\"{m['snv_abs'][0]:>6.4f}+/-{m['snv_abs'][1]:.4f} \"
          f\"{m['ood'][0]:>6.4f}+/-{m['ood'][1]:.4f}\")

print()
best_name, best_m = ranked[0]
print(f\"Best config: {best_name}\")
print(f\"  in_dist Pearson: {best_m['in_dist'][0]:.4f} +/- {best_m['in_dist'][1]:.4f}\")
print(f\"  SNV abs Pearson: {best_m['snv_abs'][0]:.4f} +/- {best_m['snv_abs'][1]:.4f}\")
print(f\"  OOD Pearson:     {best_m['ood'][0]:.4f} +/- {best_m['ood'][1]:.4f}\")
" || echo "Summary generation failed"

echo ""
echo "=== NTv3 post-trained pipeline COMPLETE — $(date) ==="
