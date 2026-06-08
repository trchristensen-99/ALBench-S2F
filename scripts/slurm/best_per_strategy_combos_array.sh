#!/bin/bash
# Best-model-per-strategy exhaustive all-subsets deploy analysis across all 9 D=30k
# Step-1 pools (3 reservoirs x 3 seeds). Companion to greedy_deploy_array.sh: same pools,
# same oracle-landscape metric, but <=14 atoms (one best model per strategy) so the full
# 2^k-1 subset lattice is enumerable. Each task writes best_per_strategy_combos.json in
# the pool's ablation/ dir; an afterok aggregate step writes the cross-pool summary.
#
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/best_per_strategy_combos_array.sh
#
#SBATCH --job-name=bps_combos
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A_%a.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-8

set -uo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TQDM_DISABLE=1

POOLS=(
  outputs/hp_step1_bakeoff/k562_dinuc_shuffle_d30000/seed42_0
  outputs/hp_step1_bakeoff/k562_dinuc_shuffle_d30000/seed43_1
  outputs/hp_step1_bakeoff/k562_dinuc_shuffle_d30000/seed44_2
  outputs/hp_step1_bakeoff/k562_genomic_d30000/seed42_0
  outputs/hp_step1_bakeoff/k562_genomic_d30000/seed43_1
  outputs/hp_step1_bakeoff/k562_genomic_d30000/seed44_2
  outputs/hp_step1_bakeoff/k562_motif_planted_v2_d30000/seed42_0
  outputs/hp_step1_bakeoff/k562_motif_planted_v2_d30000/seed43_1
  outputs/hp_step1_bakeoff/k562_motif_planted_v2_d30000/seed44_2
)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}
echo "=== bps_combos | $POOL | $(date) ==="
uv run --no-sync python scripts/analysis/best_per_strategy_combos.py --pool_dir "$POOL"
echo "=== done $(date) ==="
