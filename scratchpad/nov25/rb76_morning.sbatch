#!/bin/bash
# rb76: the morning rebuild. Scheduled to start after the overnight arms have had a night.
#
# Everything launched tonight writes cells; none of it updates a figure. Without this the
# morning starts with a directory of checkpoints and a manual list of commands to run. This
# rescores every new cell root on the eval sets that postdate them, then regenerates every
# figure that reads them, so the first thing available is pictures rather than a to-do list.
#
# `|| true` on each step on purpose: a family that has not finished should not stop the
# families that have. Each step prints what it managed, and the summary at the end counts
# cells per root so a missing figure can be traced to a missing cell rather than to a crash.
#SBATCH --job-name=rb76
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --begin=now+9hours
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/rb76_%j.log
set -uo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
export ALBENCH_EXTRA_BATTERY="$KOO" ALBENCH_SIDECAR_ROOT="$KOO/sidecars"
export ALBENCH_EXTRA_ROOTS="$KOO/outputs/mg60:$KOO/outputs/pm74:$KOO/outputs/sc77"
mkdir -p "$KOO/sidecars" /tmp/$USER/mpl /tmp/$USER/tmp

echo "=== 1. rescore new cells on the eval sets built after them ==="
for root in mg60 pm74 ps68 rp71 rp72 ac60 ac63 cm78 ly81 sc77; do
  d="$KOO/outputs/$root"
  [ -d "$d" ] || continue
  n=$(ls "$d" 2>/dev/null | wc -l)
  echo "--- $root ($n cells)"
  .venv/bin/python scratchpad/nov25/rescore_extra_sets.py \
      --dirs "$d"/*/ "$d"/*/*/ --sidecar-root "$KOO/sidecars" --max-members 3 2>&1 \
      | grep -vE 'Warning|warn' | tail -3 || true
done

echo "=== 2. the diversity analysis, over everything that now exists ==="
.venv/bin/python "$KOO/bin/diversity_analysis.py" --out "$KOO/diversity_morning.json" \
    --selections "$KOO/pools/sel_*.npz" \
    2>&1 | grep -E 'no-op|->|shard' | tail -3 || true

echo "=== 3. figures ==="
for s in scaling_mse.py scaling_branch.py mixture_figures.py make_meeting_figures.py; do
  echo "--- $s"
  .venv/bin/python "scratchpad/nov25/$s" --out outputs/figures/nov25 2>&1 \
      | grep -E 'written|Error' | tail -6 || true
done
.venv/bin/python scratchpad/nov25/genomic_branch_figure.py --emit "$KOO/genomic_branches.json" \
    2>&1 | grep -E 'endpoints|->' | tail -4 || true

echo "=== 4. what landed ==="
for root in mg60 pm74 ps68 rp71 rp72 ac60 ac61 ac63 cs62; do
  printf '%-7s %s cells\n' "$root" "$(ls "$KOO/outputs/$root" 2>/dev/null | wc -l)"
done
echo "failures logged overnight:"; cat "$KOO/watchdog_failures.log" 2>/dev/null | tail -20
echo "rb76 done"
