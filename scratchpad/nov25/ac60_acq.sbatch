#!/bin/bash
# ac60 stage 2: acquisition from the 30k GENOMIC start and the 30k MOTIF start.
#
# Every acquisition result the study has seeded from motif_shared_core or from the equal
# 7-way union. Both of the two reservoirs that matter for the recommendation are
# therefore untested as STARTING points: a rule that helps a model trained on designed
# sequence need not help one trained on real CREs, and the collaborators' first 30k will
# be real CREs if anything is.
#
# 2 starts x 3 candidate pools x 3 rules = 18 arms, all 30k -> 60k, all from a bank that
# already exists (wed41 kept weights at 30k), so no arm pays to retrain its own seed.
#
# THE RULES. `random` is the floor and is not optional -- it is the control that says
# whether a rule did anything at all, and in fig18 it came back flat (25.1/25.0/25.0/24.9
# on a 4-way union) which is exactly what a working control looks like. `epistemic` is
# the rule that won Arm B. `expected_activity` is included because fig19 showed it wins a
# different endpoint than epistemic does (designed by activity, 3.243 vs 1.940), so the
# two are not ranked, they are different objectives.
#
# --cap 800000, NOT the default. run_al scores only 200,000 candidates per round by
# default; Arm B hit that silently and never used its 1M pools, so "uncertainty beats
# random" was measured on a fifth of the candidates it was supposed to see.
#SBATCH --job-name=ac60
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ac60_%A_%a.log
#SBATCH --array=0-17%4
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/pools" "$KOO/outputs/ac60" /tmp/$USER/mpl /tmp/$USER/tmp

POOLS=(g30_own g30_u7 g30_umix m30_own m30_u7 m30_umix)
BANKS=(bench38_ln_e500_wd2_30k_c20 bench38_ln_e500_wd2_30k_c20
       bench38_ln_e500_wd2_30k_c20
       bench38_ln_e500_wd2_30k_c20_motif_shared_core
       bench38_ln_e500_wd2_30k_c20_motif_shared_core
       bench38_ln_e500_wd2_30k_c20_motif_shared_core)
METHODS=(random epistemic expected_activity)

i=$SLURM_ARRAY_TASK_ID
p=$(( i / 3 ))
m=${METHODS[$(( i % 3 ))]}
TAG="${POOLS[$p]}_${m}"
POOL="$KOO/pools/${POOLS[$p]}.npz"
BANK="$REPO/outputs/wed41/${BANKS[$p]}"

SEED=30000; STEP=30000; N=$(( SEED + STEP ))
[ -f "$POOL" ] || { echo "ac60: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ac60: seed bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ac60 $TAG: pool=$POOL bank=$BANK ($n_pt members) ${SEED}+${STEP}"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 800000 \
    --bank "$BANK" --method "$m" --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds 0=60236470,1=60236471,2=60236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/ac60/${TAG}"
echo "ac60 $TAG done"
