#!/bin/bash
# Merge sharded train caches, build val+test caches, then run HP grid search.
# Depends on sharded Enformer + Borzoi cache builds and NTv3-post cache.
#
# Submit (with dependencies):
#   /cm/shared/apps/slurm/current/bin/sbatch \
#     --dependency=afterok:<ENF_JOB>:<BOR_JOB>:<NTV3_JOB> \
#     scripts/slurm/merge_and_sweep_foundation.sh
#
#SBATCH --job-name=fm_merge_sweep
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Foundation Model Merge + Sweep — $(date) ==="
echo "Node: ${SLURMD_NODENAME}"

# ── Step 1: Merge shards ────────────────────────────────────────────────────
merge_shards() {
    local cache_dir=$1
    local n_shards=$2
    local split=$3

    local can_path="${cache_dir}/${split}_canonical.npy"
    local rc_path="${cache_dir}/${split}_rc.npy"

    # Skip if already merged
    if [ -f "$can_path" ] && [ -f "$rc_path" ]; then
        echo "  ${split}: merged cache exists — skipping."
        return
    fi

    # Check all shards exist
    for i in $(seq 0 $((n_shards - 1))); do
        if [ ! -f "${cache_dir}/${split}_shard${i}_canonical.npy" ]; then
            echo "  ERROR: ${split}_shard${i}_canonical.npy missing!"
            return 1
        fi
    done

    echo "  Merging ${split} from ${n_shards} shards..."
    uv run --no-sync python -c "
import numpy as np
from pathlib import Path
cache_dir = Path('${cache_dir}')
n_shards = ${n_shards}
split = '${split}'
shards_can, shards_rc = [], []
for i in range(n_shards):
    can = np.load(cache_dir / f'{split}_shard{i}_canonical.npy')
    rc = np.load(cache_dir / f'{split}_shard{i}_rc.npy')
    shards_can.append(can)
    shards_rc.append(rc)
    print(f'    shard {i}: {can.shape}')
merged_can = np.concatenate(shards_can, axis=0)
merged_rc = np.concatenate(shards_rc, axis=0)
np.save(cache_dir / f'{split}_canonical.npy', merged_can)
np.save(cache_dir / f'{split}_rc.npy', merged_rc)
print(f'    Merged: {merged_can.shape} ({merged_can.dtype})')
# Clean up shard files
for i in range(n_shards):
    (cache_dir / f'{split}_shard{i}_canonical.npy').unlink()
    (cache_dir / f'{split}_shard{i}_rc.npy').unlink()
print('    Shard files deleted.')
"
}

ENF_CACHE="outputs/enformer_k562_cached/embedding_cache"
BOR_CACHE="outputs/borzoi_k562_cached/embedding_cache"

echo ""
echo "--- Merging Enformer train shards ---"
merge_shards "$ENF_CACHE" 4 train

echo ""
echo "--- Merging Borzoi train shards ---"
merge_shards "$BOR_CACHE" 4 train

# ── Step 2: Build val + test caches (sequential, ~1-2h each) ───────────────
echo ""
echo "--- Building Enformer val + test caches ---"
if [ -f "$ENF_CACHE/train_canonical.npy" ] && [ ! -f "$ENF_CACHE/val_canonical.npy" ]; then
    uv run --no-sync python scripts/build_enformer_embedding_cache.py \
        --data-path data/k562 \
        --cache-dir "$ENF_CACHE" \
        --splits val \
        --include-test \
        --batch-size 4
else
    echo "  val+test already exist or train missing — skipping."
fi

echo ""
echo "--- Building Borzoi val + test caches ---"
if [ -f "$BOR_CACHE/train_canonical.npy" ] && [ ! -f "$BOR_CACHE/val_canonical.npy" ]; then
    uv run --no-sync python scripts/build_borzoi_embedding_cache.py \
        --data-path data/k562 \
        --cache-dir "$BOR_CACHE" \
        --splits val \
        --include-test \
        --batch-size 2
else
    echo "  val+test already exist or train missing — skipping."
fi

# ── Step 3: HP grid search for all 3 models ────────────────────────────────
echo ""
echo "=== HP Grid Search — $(date) ==="

LRS=(0.0001 0.0005 0.001)
WDS=(0.000001 0.0001 0.001)
DROPOUTS=(0.1 0.3 0.5)

for MODEL_CFG in \
    "enformer ${ENF_CACHE} 3072" \
    "borzoi ${BOR_CACHE} 1536" \
    "ntv3_post outputs/ntv3_post_k562_cached/embedding_cache 1536"; do

    MODEL=$(echo "$MODEL_CFG" | cut -d' ' -f1)
    CACHE=$(echo "$MODEL_CFG" | cut -d' ' -f2)
    DIM=$(echo "$MODEL_CFG" | cut -d' ' -f3)
    OUT_BASE="outputs/foundation_grid_search/${MODEL}"

    if [ ! -f "${CACHE}/train_canonical.npy" ]; then
        echo "  ${MODEL}: train cache missing — skipping."
        continue
    fi

    echo ""
    echo "--- ${MODEL} sweep (embed_dim=${DIM}) ---"
    TOTAL=$(( ${#LRS[@]} * ${#WDS[@]} * ${#DROPOUTS[@]} ))
    COUNT=0

    for lr in "${LRS[@]}"; do
      for wd in "${WDS[@]}"; do
        for do_val in "${DROPOUTS[@]}"; do
          COUNT=$((COUNT + 1))
          OUT_DIR="${OUT_BASE}/lr${lr}_wd${wd}_do${do_val}"

          if ls "${OUT_DIR}"/seed_*/result.json > /dev/null 2>&1; then
              echo "[${COUNT}/${TOTAL}] SKIP: lr=${lr} wd=${wd} do=${do_val}"
              continue
          fi

          echo "[${COUNT}/${TOTAL}] ${MODEL}: lr=${lr} wd=${wd} dropout=${do_val} — $(date)"
          uv run --no-sync python experiments/train_foundation_cached.py \
              ++model_name="${MODEL}" \
              ++cache_dir="${CACHE}" \
              ++embed_dim="${DIM}" \
              ++output_dir="${OUT_DIR}" \
              ++lr="${lr}" \
              ++weight_decay="${wd}" \
              ++dropout="${do_val}" \
              ++seed=42 \
              || echo "FAILED: ${MODEL} lr=${lr} wd=${wd} dropout=${do_val}"
        done
      done
    done

    # Summary
    echo ""
    echo "=== ${MODEL} GRID SEARCH SUMMARY ==="
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
if results:
    best = results[0]
    print(f\"\nBest: {best['config']} (in_dist={best['in_dist']:.4f})\")
" || echo "Summary generation failed"

    echo "--- ${MODEL} done — $(date) ---"
done

echo ""
echo "=== All done — $(date) ==="
