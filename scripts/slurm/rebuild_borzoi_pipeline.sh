#!/bin/bash
# Rebuild Borzoi embedding cache with patched source (fixed fast_relative_shift)
# then retrain head on corrected embeddings. 3 random seeds.
#
# The original cache used a broken as_strided-based relative position encoding.
# The patched source uses torch.gather for correct position indexing.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/rebuild_borzoi_pipeline.sh
#
#SBATCH --job-name=borzoi_rebuild
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

CACHE_DIR="outputs/borzoi_k562_cached_v2/embedding_cache"
OUTPUT_DIR="outputs/borzoi_k562_cached_v2"

echo "=== Phase 1: Build Borzoi embedding cache (patched source) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"
echo "Cache dir: ${CACHE_DIR}"

uv run --no-sync python scripts/build_borzoi_embedding_cache.py \
    --data-path data/k562 \
    --cache-dir "${CACHE_DIR}" \
    --splits train val \
    --include-test \
    --batch-size 2

echo "Cache build DONE — $(date)"

# Validate: check for NaN in cached embeddings
echo ""
echo "=== Validating cache (checking for NaN) ==="
uv run --no-sync python -c "
import numpy as np
from pathlib import Path

cache_dir = Path('${CACHE_DIR}')
all_ok = True
for npy_file in sorted(cache_dir.glob('*.npy')):
    arr = np.load(str(npy_file))
    n_nan = np.isnan(arr).sum()
    n_inf = np.isinf(arr).sum()
    status = 'OK' if (n_nan == 0 and n_inf == 0) else 'FAIL'
    if status == 'FAIL':
        all_ok = False
    print(f'  {npy_file.name:35s} shape={str(arr.shape):20s} NaN={n_nan:6d} Inf={n_inf:6d} [{status}]')

if not all_ok:
    print('ERROR: Cache contains NaN/Inf values!')
    exit(1)
print('All cache files validated OK.')
"

echo ""
echo "=== Phase 2: Train head on corrected embeddings (3 seeds) ==="
echo "Start: $(date)"

for seed_idx in 0 1 2; do
    echo ""
    echo "--- Seed ${seed_idx} ---"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name=borzoi \
        ++cache_dir="${CACHE_DIR}" \
        ++embed_dim=1536 \
        ++output_dir="${OUTPUT_DIR}"
    echo "Seed ${seed_idx} DONE — $(date)"
done

echo ""
echo "=== Pipeline DONE — $(date) ==="

# Print summary
uv run --no-sync python -c "
import json
from pathlib import Path
import numpy as np

base = Path('${OUTPUT_DIR}')
results = []
for rfile in sorted(base.glob('seed_*/result.json')):
    r = json.load(open(rfile))
    tm = r.get('test_metrics', {})
    results.append({
        'seed': rfile.parent.name,
        'in_dist': tm.get('in_distribution', {}).get('pearson_r', 0),
        'snv_abs': tm.get('snv_abs', {}).get('pearson_r', 0),
        'ood': tm.get('ood', {}).get('pearson_r', 0),
    })

print(f\"\"\"
Borzoi v2 Results (patched fast_relative_shift):
{'Seed':<12} {'InDist':>10} {'SNV':>10} {'OOD':>10}\"\"\")
for r in results:
    print(f\"{r['seed']:<12} {r['in_dist']:>10.4f} {r['snv_abs']:>10.4f} {r['ood']:>10.4f}\")
if len(results) > 1:
    ids = [r['in_dist'] for r in results]
    snvs = [r['snv_abs'] for r in results]
    oods = [r['ood'] for r in results]
    print(f\"\"\"{'Mean±Std':<12} {np.mean(ids):>5.4f}±{np.std(ids):.4f} {np.mean(snvs):>5.4f}±{np.std(snvs):.4f} {np.mean(oods):>5.4f}±{np.std(oods):.4f}\"\"\")
" || echo "Summary generation failed"
