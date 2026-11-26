#!/bin/bash
# rp79: two more selection seeds, on the `fast` QOS tier nothing else is using.
#
# The three QOS tiers have independent GPU caps (fast 2, default 4, slow_nice 20 per type),
# and tonight everything sits in slow_nice, which is now at its 20-V100 ceiling while the
# cluster still has V100s free. These are the same arms rp71 runs, at selection seeds 3 and
# 4, placed in `fast` purely because that tier's allocation is otherwise idle.
#
# Five selection seeds rather than three matters for the specific claim being made: the
# gaps between rules are of order 0.02 in MSE, and three draws give a standard error with
# two degrees of freedom. If the rule differences are to survive a registered report they
# need to be larger than a spread estimated on more than three points.
#
# fast has a 4-hour wall. One arm is a selection pass over 540,000 candidates plus three
# seeds at 30k -> 60k, which is about an hour on a V100, so every task has several times
# the wall it needs.
#SBATCH --job-name=rp79
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/rp79_%A_%a.log
#SBATCH --array=0-5%2
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/rp71" /tmp/$USER/mpl /tmp/$USER/tmp

METHODS=(random epistemic coverage_epistemic)
SEEDS=(3 4)
BANK="$REPO/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
POOL="$KOO/pools/u6_540000.npz"
SEED=30000; STEP=30000; N=60000

i=$SLURM_ARRAY_TASK_ID
M=${METHODS[$(( i % 3 ))]}
S=${SEEDS[$(( i / 3 ))]}
TAG="u6_540k_${M}_s${S}"

[ -f "$POOL" ] || { echo "rp79: candidate pool missing: $POOL"; exit 1; }
echo "rp79 $TAG: method=$M selection-seed=$S"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 540000 \
    --bank "$BANK" --method "$M" --select "$STEP" --seed "$S" --out "$NEXT"
fi

# Same training seeds as rp71, so the five selection seeds differ in selection alone and
# the spread across them is not contaminated by a different training draw.
.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds 0=71236470,1=71236471,2=71236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/rp71/${TAG}"
echo "rp79 $TAG done"
