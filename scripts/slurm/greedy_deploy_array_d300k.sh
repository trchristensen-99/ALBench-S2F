#!/bin/bash
# Greedy per-model deploy-pool selection across all 9 D=300k Step-1 pools
# (3 reservoirs x 3 seeds). Mirror of greedy_deploy_array.sh for the d300000 tier.
# Each task writes greedy_deploy.json in the pool's ablation/ dir; finalize then
# aggregates across both D tiers to one global N*. CPU-only.
#
# Auto-fired by watchdog_step1_bakeoff_d300k.sh once the bake-off completes; can
# also be submitted manually:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/greedy_deploy_array_d300k.sh
#
#SBATCH --job-name=greedy_deploy_d300k
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A_%a.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A_%a.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-8

set -uo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TQDM_DISABLE=1

POOLS=(
  outputs/hp_step1_bakeoff/k562_dinuc_shuffle_d300000/seed42_0
  outputs/hp_step1_bakeoff/k562_dinuc_shuffle_d300000/seed43_1
  outputs/hp_step1_bakeoff/k562_dinuc_shuffle_d300000/seed44_2
  outputs/hp_step1_bakeoff/k562_genomic_d300000/seed42_0
  outputs/hp_step1_bakeoff/k562_genomic_d300000/seed43_1
  outputs/hp_step1_bakeoff/k562_genomic_d300000/seed44_2
  outputs/hp_step1_bakeoff/k562_motif_planted_v2_d300000/seed42_0
  outputs/hp_step1_bakeoff/k562_motif_planted_v2_d300000/seed43_1
  outputs/hp_step1_bakeoff/k562_motif_planted_v2_d300000/seed44_2
)
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}
echo "=== greedy_deploy_d300k | $POOL | $(date) ==="
uv run --no-sync python scripts/analysis/greedy_deploy_select.py \
    --pool_dir "$POOL" --max_n 16 --prefilter 120
echo "=== done $(date) ==="
