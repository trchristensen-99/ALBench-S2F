#!/bin/bash
# Submit 384-bp compact-window training jobs (from HPC repo root).
# Builds the shared embedding cache first; training jobs depend on it.
#
# Usage (on HPC):
#   bash scripts/slurm/submit_compact_jobs.sh
#
# Also submits test-set cache build and a new chrom-test eval job after
# the training jobs complete so results land automatically.

set -e
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

echo "=== Compact-window (384bp) job submission ==="

# 1. Build shared compact cache (train + val)
cache_jid=$($SBATCH --parsable scripts/slurm/build_compact_cache.sh)
echo "  Cache build job: $cache_jid"

# 2. Train all 4 T-agnostic heads once cache is ready
for head in sum mean max center; do
    jid=$($SBATCH --dependency=afterok:${cache_jid} \
        scripts/slurm/train_oracle_alphagenome_full_${head}_compact.sh)
    echo "  ag_${head}_compact: $jid (depends on $cache_jid)"
done
