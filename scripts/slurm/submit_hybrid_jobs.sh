#!/bin/bash
# Run from repo root on HPC after syncing (e.g. cd /grid/wsbs/home_norepl/christen/ALBench-S2F).
# Ensures uv env has deps then submits all hybrid training jobs.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:$PYTHONPATH"

echo "Syncing uv environment (jax, jaxlib, optax)..."
uv sync

echo "Submitting hybrid jobs..."
for script in scripts/slurm/train_oracle_alphagenome_full_sum_hybrid.sh \
              scripts/slurm/train_oracle_alphagenome_full_mean_hybrid.sh \
              scripts/slurm/train_oracle_alphagenome_full_max_hybrid.sh \
              scripts/slurm/train_oracle_alphagenome_full_center_hybrid.sh \
              scripts/slurm/train_oracle_alphagenome_full_flatten_hybrid.sh; do
  sbatch "$script" || true
done
echo "Done. Check: squeue -u \$USER"
