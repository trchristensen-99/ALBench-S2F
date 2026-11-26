#!/bin/bash
# ps68 stage 2: how pool size and selection rule interact, with genomic out of the pool.
#
# Twelve arms: four candidate-pool sizes crossed with three rules, every one starting from
# the same 30,000 real CREs and selecting the same 30,000 more. The only things that vary
# are how much there was to choose from and who chose.
#
# WHAT `random` IS HERE, and it is worth saying in the legend rather than leaving implicit:
# a random draw from an even six-reservoir union IS the naive mixture corpus. So the random
# line is not only the floor for the rules -- it is also the "just mix them" arm, and it
# should sit near the average of the six single-reservoir curves. If a rule cannot beat it,
# the rule is not buying anything a blender would not.
#
# `diversity_kmeans` is included because a pool-size sweep is exactly where a diversity
# rule should come into its own: with little to choose from there is nothing to be diverse
# about, and if it never separates from random at any size that is a real result about the
# rule rather than about the pool. Its embedding step is the memory-hungry one, so the
# largest pool is capped for it alone and the cap is recorded in the arm's meta.
#SBATCH --job-name=ps68
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ps68_%A_%a.log
#SBATCH --array=0-11%4
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/ps68" /tmp/$USER/mpl /tmp/$USER/tmp

SIZES=(60000 180000 540000 1500000)
METHODS=(random epistemic diversity_kmeans)
BANK="$REPO/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
SEED=30000; STEP=30000; N=60000

i=$SLURM_ARRAY_TASK_ID
SZ=${SIZES[$(( i / 3 ))]}
M=${METHODS[$(( i % 3 ))]}
TAG="u6_${SZ}_${M}"
POOL="$KOO/pools/u6_${SZ}.npz"

[ -f "$POOL" ] || { echo "ps68: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ps68: seed bank $BANK has $n_pt members, need >=3"; exit 1; }

# Score the WHOLE pool, not run_al's 200,000 default -- the default is what made an earlier
# arm family silently use a fifth of its candidates, and here the pool size IS the variable,
# so a cap below it would flatten the very axis this job exists to measure.
CAP=$SZ
if [ "$M" = "diversity_kmeans" ] && [ "$SZ" -gt 600000 ]; then CAP=600000; fi
echo "ps68 $TAG: pool=$SZ method=$M cap=$CAP"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap "$CAP" \
    --bank "$BANK" --method "$M" --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds 0=68236470,1=68236471,2=68236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/ps68/${TAG}"
echo "ps68 $TAG done"
