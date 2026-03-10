#!/bin/bash
# Build NTv3 500M embedding cache + train head.
# Uses the larger 500M_multi_species_v2 model (1024D embeddings, 29 layers)
# vs the default 250M (768D, 24 layers) for richer representations.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/nt_500m_pipeline.sh
#
#SBATCH --job-name=nt_500m
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

CACHE_DIR="outputs/nt_k562_500m/embedding_cache"
OUTPUT_DIR="outputs/nt_k562_500m"

echo "=== Phase 1: Build NTv3 500M cache (1024D) ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python scripts/build_nt_embedding_cache.py \
    --data-path data/k562 \
    --cache-dir "${CACHE_DIR}" \
    --splits train val \
    --include-test \
    --batch-size 32 \
    --model-name 500M_multi_species_v2 \
    --embedding-layer 29 \
    --max-positions 256

echo "Cache build DONE — $(date)"

echo ""
echo "=== Phase 2: Train head on 1024D embeddings (3 seeds) ==="
for seed_idx in 0 1 2; do
    echo "--- Seed ${seed_idx} ---"
    uv run --no-sync python experiments/train_foundation_cached.py \
        ++model_name=nt_500m \
        ++cache_dir="${CACHE_DIR}" \
        ++embed_dim=1024 \
        ++output_dir="${OUTPUT_DIR}"
    echo "Seed ${seed_idx} DONE — $(date)"
done

echo ""
echo "=== Pipeline DONE — $(date) ==="

# Print summary
uv run --no-sync python -c "
import json, numpy as np
from pathlib import Path

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

print(f\"\nNTv3 500M Results:\")
print(f\"{'Seed':<12} {'InDist':>10} {'SNV':>10} {'OOD':>10}\")
for r in results:
    print(f\"{r['seed']:<12} {r['in_dist']:>10.4f} {r['snv_abs']:>10.4f} {r['ood']:>10.4f}\")
if len(results) > 1:
    ids = [r['in_dist'] for r in results]
    snvs = [r['snv_abs'] for r in results]
    oods = [r['ood'] for r in results]
    print(f\"{'Mean±Std':<12} {np.mean(ids):>.4f}±{np.std(ids):.4f} {np.mean(snvs):>.4f}±{np.std(snvs):.4f} {np.mean(oods):>.4f}±{np.std(oods):.4f}\")
" || echo "Summary generation failed"
