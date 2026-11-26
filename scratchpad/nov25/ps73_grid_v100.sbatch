#!/bin/bash
# ps73: the rest of the pool-size x rule grid, on the cards that are actually free.
#
# ps68 ran four candidate-pool sizes against random / uncertainty / diversity. This adds
# the three rules it did not have -- the two coverage rules and activity_normalized -- so
# the grid is complete and the pool-size axis can be read for every rule rather than for
# three of six. Completing it matters because the size axis may not act the same way on a
# rule that ranks globally and a rule that allocates by stratum: a bigger pool gives a
# ranking rule more extremes to find, but gives a coverage rule only more of the same.
#
# V100, DELIBERATELY. The h100 partition is at 72 of 76 allocated cluster-wide while 46 of
# 56 V100s sit idle. A V100 is roughly 42% of an H100 here, so a 12-arm grid that would
# queue for hours behind contended H100s finishes overnight on cards nobody is using. For
# an overnight batch the right question is throughput, not per-card speed.
#SBATCH --job-name=ps73
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ps73_%A_%a.log
#SBATCH --array=0-11%12
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/ps68" /tmp/$USER/mpl /tmp/$USER/tmp

SIZES=(60000 180000 540000 1500000)
METHODS=(coverage_random coverage_epistemic activity_normalized)
BANK="$REPO/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
SEED=30000; STEP=30000; N=60000

i=$SLURM_ARRAY_TASK_ID
SZ=${SIZES[$(( i / 3 ))]}
M=${METHODS[$(( i % 3 ))]}
TAG="u6_${SZ}_${M}"
POOL="$KOO/pools/u6_${SZ}.npz"

[ -f "$POOL" ] || { echo "ps73: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ps73: seed bank $BANK has $n_pt members, need >=3"; exit 1; }

# Score the whole pool: the pool size IS the variable here, so a cap below it would flatten
# the axis the job exists to measure. The coverage rules need the full predicted-activity
# distribution to cut deciles from, so a subsample would also move their strata.
echo "ps73 $TAG: pool=$SZ method=$M (writing into the ps68 tree so the grid reads as one)"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap "$SZ" \
    --bank "$BANK" --method "$M" --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds 0=68236470,1=68236471,2=68236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/ps68/${TAG}"
echo "ps73 $TAG done"
