#!/bin/bash
# rp71: replicate SELECTION seeds, and the two coverage rules.
#
# Every acquisition arm the study has ever run used --seed 0. So we have measured training
# -seed variance (three seeds per cell, stacked) and never once measured SELECTION variance
# -- how much of "uncertainty beats random by 0.02" is the rule, and how much is which
# 30,000 of a 540,000-row pool it happened to land on. Without that number the comparison
# between rules has no error bar at all, which is the right thing to be sceptical about.
#
# Six rules x three selection seeds, one pool, one start. The three rules ps68 already ran
# at seed 0 are deliberately repeated here under the same seed: same rule, same pool, same
# seed, a different job, so the selections must come out identical. That is a free
# determinism check on the whole acquisition path.
#
# THE TWO NEW RULES ARE A PAIR, and the pure one matters more:
#   coverage_random     equal allocation across predicted-activity deciles, random inside
#                       each. No uncertainty anywhere in it. This is the version that can
#                       be written down as a recipe and ordered without running active
#                       learning at all.
#   coverage_epistemic  the same allocation, most-uncertain inside each decile.
# If coverage_random matches uncertainty, the useful output is a recipe rather than an
# algorithm. If it does not, the gap is what the model knows that the summary statistics
# do not -- and either answer is worth more than another ranking of rules.
#SBATCH --job-name=rp71
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/rp71_%A_%a.log
#SBATCH --array=0-17%4
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/rp71" /tmp/$USER/mpl /tmp/$USER/tmp

METHODS=(random epistemic diversity_kmeans coverage_random coverage_epistemic activity_normalized)
SEEDS=(0 1 2)
BANK="$REPO/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
POOL="$KOO/pools/u6_540000.npz"     # the mid rung; ps68 says whether this is the right size
SEED=30000; STEP=30000; N=60000

i=$SLURM_ARRAY_TASK_ID
M=${METHODS[$(( i % 6 ))]}
S=${SEEDS[$(( i / 6 ))]}
TAG="u6_540k_${M}_s${S}"

[ -f "$POOL" ] || { echo "rp71: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "rp71: seed bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "rp71 $TAG: method=$M selection-seed=$S"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 540000 \
    --bank "$BANK" --method "$M" --select "$STEP" --seed "$S" --out "$NEXT"
fi

# Training seeds are held FIXED across the three selection seeds, so the spread between
# replicates is selection variance alone rather than the two sources mixed together.
.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds 0=71236470,1=71236471,2=71236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/rp71/${TAG}"
echo "rp71 $TAG done"
