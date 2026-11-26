#!/bin/bash
# ac66: the genomic-start arms, moved onto the fast H100 queue.
#
# The scaling-law figures are being cut down to ONE starting corpus -- 30k of real CREs --
# so that the only thing differing between curves is the rule and the pool it chooses from.
# That makes these seven cells the figure's entire content, and they were sitting behind
# twelve motif-start cells on a preemptible queue. Here they are the whole job.
#
# THE SEVENTH CELL IS THE ONE THAT MATTERS MOST. `g60_scratch` is 60,000 rows of genomic's
# own nested prefix with no selection at all. Without it every branch in the figure is
# measured against nothing, and "uncertainty reaches X at 60k" cannot be separated from
# "30,000 more real CREs reach X at 60k". It is the cheapest cell here and the only one
# that can make the others mean anything.
#
# qos=fast is 2 H100 and a 4-hour wall. One cell is a selection pass (~5 min over 800k
# candidates) plus three seeds at ~12 min, so every task has about five times the wall it
# needs; nothing here is at risk of being cut off mid-training.
#SBATCH --job-name=ac66
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ac66_%A_%a.log
#SBATCH --array=0-6%2
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/pools" "$KOO/outputs/ac60" /tmp/$USER/mpl /tmp/$USER/tmp

BANK="$REPO/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
NEST="$REPO/outputs/grid_pools/k562_nested"
SEED=30000; STEP=30000; N=60000
SEEDS="0=60236470,1=60236471,2=60236472"

# Tasks 0-5: the two union pools under the three rules. g30_own was already finished on
# the slow queue, so it is not repeated here.
POOLS=(g30_u7 g30_u7 g30_u7 g30_umix g30_umix g30_umix)
METHODS=(random epistemic expected_activity random epistemic expected_activity)

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -eq 6 ]; then
  TAG=g60_scratch
  echo "ac66 $TAG: 60,000 rows of genomic's own prefix, no selection"
  .venv/bin/python scripts/program_oct22/bench33_legnet.py \
    --D "$N" --cfg e500_wd2 --seeds "$SEEDS" --ctx 20 --save-weights \
    --pool "$KOO/pools/genomic_d60000.npz" --out-root "$KOO/outputs/ac60/${TAG}"
  echo "ac66 $TAG done"; exit 0
fi

TAG="${POOLS[$i]}_${METHODS[$i]}"
POOL="$KOO/pools/${POOLS[$i]}.npz"
[ -f "$POOL" ] || { echo "ac66: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ac66: seed bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ac66 $TAG: pool=$(basename "$POOL") method=${METHODS[$i]}"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 800000 \
    --bank "$BANK" --method "${METHODS[$i]}" --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds "$SEEDS" --ctx 20 --save-weights \
  --pool "$NEXT" --out-root "$KOO/outputs/ac60/${TAG}"
echo "ac66 $TAG done"
