#!/bin/bash
#SBATCH --job-name=abl_robust
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%A.out
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --export=ALL
set -uo pipefail
set +u
source /etc/profile.d/modules.sh
[ -f ~/.bash_profile ] && source ~/.bash_profile
set -u
module load EB5 2>/dev/null || true
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/experiments"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

ABL="scripts/analysis/strategy_combination_ablation.py"

echo "=== ROBUSTNESS: completed hp_search (reservoir x D) cells (mixed6) ==="
for cell in outputs/hp_search/k562_*_d*; do
  [ -d "$cell" ] || continue
  ok=1
  for v in algo llm_default llm_diverse llm_exploit; do
    ls "$cell/$v"/r*.npz >/dev/null 2>&1 || ok=0
  done
  if [ "$ok" != 1 ]; then echo "skip $cell (incomplete variants)"; continue; fi
  echo "--- ablation: $cell ---"
  uv run --no-sync python "$ABL" \
    --pool_dir "$cell" --out_dir "$cell/ablation" \
    --n_boot_fit 150 --n_boot_test 500 --seed 0 \
    --budget_grid 6,12,24,48 || echo "FAILED $cell"
done

echo "=== abl_robust job done ==="
