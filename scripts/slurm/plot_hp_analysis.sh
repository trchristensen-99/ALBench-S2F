#!/bin/bash
#SBATCH --job-name=hp_analysis
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/hp_analysis_%j.out
#SBATCH --error=logs/hp_analysis_%j.err

set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p logs outputs/analysis_figures

echo "== 1) Re-ablate D=1M cells (snapshot of current pools) =="
for cell in outputs/hp_search/*_d1000000; do
  [ -d "$cell" ] || continue
  echo "-- ablating $cell"
  uv run --no-sync python scripts/analysis/strategy_combination_ablation.py \
    --pool_dir "$cell" --out_dir "$cell/ablation" \
    --n_boot_fit 150 --n_boot_test 500 || echo "   (skip: $cell ablation failed)"
done

echo "== 2) Aggregate 50-round curves + build all figures =="
uv run --no-sync python scripts/analysis/plot_hp_rounds_and_ensemble.py \
  --rounds_root outputs/hp_rounds_scaling \
  --phase2_rounds_summary outputs/phase2_longrounds/D30000/rounds_summary.json \
  --hp_search outputs/hp_search \
  --fig_dir outputs/analysis_figures

echo "== DONE =="
ls -la outputs/analysis_figures/
