#!/bin/bash
# ac61: the 300k start, which is the setting the collaborators said is closest to the run.
#
# ab42 already measured 300k -> 450k for random / epistemic / badge_epistemic_only on the
# two motif reservoirs. Three things it did not do, and each changes what the arm means:
#
#   expected_activity  -- fig19 showed epistemic and expected_activity win DIFFERENT
#                         endpoints, so "which rule at 300k" has no answer until both ran.
#   a union pool       -- ab42 let each rule choose only from its own reservoir. A 4M
#                         order draws from several generators, so the realistic question
#                         is what a rule does with all of them at this size.
#   a matched control  -- 450k of motif_shared_core's own nested prefix, no acquisition.
#                         Without it, "acquisition reaches 0.88" cannot be separated from
#                         "150k more sequences reaches 0.88", which is the only comparison
#                         that decides anything.
#
# genomic is absent on purpose and the absence is a result: 300k seed rows exhaust the
# ~315k distinct real CREs, so a genomic 300k -> 450k arm cannot be constructed. That is
# the quantitative form of "a 4M library cannot be mostly real sequence".
#SBATCH --job-name=ac61
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ac61_%A_%a.log
#SBATCH --array=0-3%2
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/pools" "$KOO/outputs/ac61" /tmp/$USER/mpl /tmp/$USER/tmp

SEED=300000; STEP=150000; N=450000
BANK="$REPO/outputs/wed41/bench38_ln_e300_wd05_300k_c20_motif_shared_core"
NEST="$REPO/outputs/grid_pools/k562_nested"
SEEDS="0=45236470,1=45236471,2=45236472"

i=$SLURM_ARRAY_TASK_ID
case $i in
  0) TAG=m300_own_expected_activity; POOL="$NEST/motif_shared_core_d1000000.npz"; M=expected_activity ;;
  1) TAG=m300_u7_epistemic;          POOL="$KOO/pools/m300_u7.npz";               M=epistemic ;;
  2) TAG=m300_u7_random;             POOL="$KOO/pools/m300_u7.npz";               M=random ;;
  3) TAG=m450_scratch;               POOL="$KOO/pools/motif_shared_core_d450000.npz";  M="" ;;
esac

if [ -z "$M" ]; then
  # The control trains on the reservoir's own nested prefix: 450k rows of exactly the data
  # an arm would have had WITHOUT choosing. --D truncates to the prefix, which is what
  # makes it the matched comparison rather than a differently-composed 450k.
  echo "ac61 $TAG: from-scratch control, first ${N} rows of $(basename "$POOL")"
  .venv/bin/python scripts/program_oct22/bench33_legnet.py \
    --D "$N" --cfg e300_wd05 --seeds "$SEEDS" --ctx 20 --save-weights \
    --pool "$POOL" --out-root "$KOO/outputs/ac61/${TAG}"
  echo "ac61 $TAG done"; exit 0
fi

[ -f "$POOL" ] || { echo "ac61: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ac61: seed bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ac61 $TAG: pool=$POOL method=$M ${SEED}+${STEP}"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 800000 \
    --bank "$BANK" --method "$M" --select "$STEP" --seed 0 --out "$NEXT"
fi

# e300_wd05 is the 300k row of the tuned table and 450k is one step past it. State that
# with the numbers: the between-method differences are safe, the absolute level is one
# step of extrapolation.
.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e300_wd05 --seeds "$SEEDS" --ctx 20 --save-weights \
  --pool "$NEXT" --out-root "$KOO/outputs/ac61/${TAG}"
echo "ac61 $TAG done"
