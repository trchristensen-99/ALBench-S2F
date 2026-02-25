#!/bin/bash
# Submit all 5 AlphaGenome Boda heads with full shift augmentation (hybrid mode).
# Hybrid: 50% of batches use precomputed canonical/RC cache; 50% run encoder with shift+RC.
# Reuses existing cache at outputs/ag_flatten/embedding_cache (no rebuild).
# Outputs go to outputs/ag_*_hybrid so no_shift runs are not overwritten.
#
# Usage (on HPC): cd /grid/wsbs/home_norepl/christen/ALBench-S2F && bash scripts/slurm/train_oracle_alphagenome_full_hybrid.sh
# Or submit individually: sbatch scripts/slurm/train_oracle_alphagenome_full_sum_hybrid.sh

set -e
cd "$(dirname "$0")/../.." || exit 1

sbatch scripts/slurm/train_oracle_alphagenome_full_sum_hybrid.sh
sbatch scripts/slurm/train_oracle_alphagenome_full_mean_hybrid.sh
sbatch scripts/slurm/train_oracle_alphagenome_full_max_hybrid.sh
sbatch scripts/slurm/train_oracle_alphagenome_full_center_hybrid.sh
sbatch scripts/slurm/train_oracle_alphagenome_full_flatten_hybrid.sh
echo "Submitted 5 hybrid (full-shift) jobs. Check: squeue -u christen"
