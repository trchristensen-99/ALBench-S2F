#!/bin/bash
# Step-1 bake-off recipe analysis (Part A: GPU-seconds/matched-budget knee -> N*;
# Part B: ElasticNet ensemble recipe via control-ladder + forward-selection + LOO).
#
# Runs strategy_combination_ablation.py per (cell x seed) on the COMPLETED D=30k
# bake-off. Each seed dir has its OWN val split (data_seed varies the 10% holdout),
# so the ablation MUST run per (cell x seed) -- never pooled across seeds.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/recipe_analysis_step1.sh
#
#SBATCH --job-name=s1_recipe
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err

set -uo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TQDM_DISABLE=1
mkdir -p logs

# Which D tier(s) to analyze. Default 30000; pass STEP1_DS=300000 once that tier
# completes to analyze it too (cells are namespaced k562_*_d<D>).
DS="${STEP1_DS:-30000}"

n_done=0 n_fail=0 n_skip=0
for D in ${DS//,/ }; do
  for cell in outputs/hp_step1_bakeoff/k562_*_d${D}; do
    [ -d "$cell" ] || continue
    for seed in "$cell"/seed*; do
      [ -d "$seed" ] || continue
      nmeta=$(find "$seed" -name "*_meta.json" | wc -l)
      if [ "$nmeta" -lt 100 ]; then
        echo "SKIP (only $nmeta meta): $seed"; n_skip=$((n_skip+1)); continue
      fi
      out="$seed/ablation"
      if [ -f "$out/ablation_report.json" ]; then
        echo "SKIP (already done): $seed"; n_skip=$((n_skip+1)); continue
      fi
      echo "== ablating $seed ($nmeta meta) =="
      if uv run --no-sync python scripts/analysis/strategy_combination_ablation.py \
          --pool_dir "$seed" --out_dir "$out" \
          --n_boot_fit 150 --n_boot_test 500; then
        n_done=$((n_done+1))
      else
        echo "   (FAILED: $seed)"; n_fail=$((n_fail+1))
      fi
    done
  done
done

echo "=== recipe analysis complete: done=$n_done failed=$n_fail skipped=$n_skip ==="
find outputs/hp_step1_bakeoff -name ablation_report.json | sort
