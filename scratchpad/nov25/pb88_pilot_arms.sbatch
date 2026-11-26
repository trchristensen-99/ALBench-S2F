#!/bin/bash
# pb88: every rule, run from three independent pilots, with independent training seeds.
#
# This is the replication unit that actually carries error for a deterministic rule. Each
# of the three replicates is an independent draw of everything downstream of the corpus:
# its own pilot weights (pb87), its own selection, and its own three training seeds. A
# replicate shares nothing with another except the data it starts from and the pool it
# chooses from, which is what makes the spread across them an honest standard error rather
# than a measure of GPU non-determinism.
#
# Six rules x three pilots = 18 arms. The same six as rp71, so last night's seed-replicated
# numbers and these pilot-replicated ones sit on the same axes and can be read together --
# and the comparison between the two spreads is itself the result: if the pilot spread is
# much larger, every rule gap quoted so far has been quoted against the wrong noise floor.
#
# diversity_kmeans runs at main.1 rather than the penultimate layer, because the penultimate
# embedding is ~1-D along predicted activity (R^2 1.000, 1.13 effective dimensions) and a
# distance rule there re-ranks by activity. main.1 keeps 21 effective dimensions at R^2 0.70.
#SBATCH --job-name=pb88
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/pb88_%A_%a.log
#SBATCH --array=0-17%18
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/pb88" /tmp/$USER/mpl /tmp/$USER/tmp

METHODS=(random epistemic coverage_random coverage_epistemic diversity_kmeans activity_normalized)
PILOTS=(A B C)
# One training-seed triple per pilot, so a replicate is independent end to end.
TRAIN_A="0=88100001,1=88100002,2=88100003"
TRAIN_B="0=88200001,1=88200002,2=88200003"
TRAIN_C="0=88300001,1=88300002,2=88300003"

POOL="$KOO/pools/u6_540000.npz"
SEED=30000; STEP=30000; N=60000

i=$SLURM_ARRAY_TASK_ID
M=${METHODS[$(( i % 6 ))]}
P=${PILOTS[$(( i / 6 ))]}
case $P in
  A) TRAIN=$TRAIN_A ;;
  B) TRAIN=$TRAIN_B ;;
  C) TRAIN=$TRAIN_C ;;
esac
BANK="$KOO/outputs/pb87/pilot${P}/bench38_ln_e500_wd2_30k_c20"
TAG="u6_540k_${M}_pilot${P}"

EXTRA=""
if [ "$M" = "diversity_kmeans" ]; then
  EXTRA="--embed-layer main.1"
  TAG="${TAG}_Lmain_1"
fi

[ -f "$POOL" ] || { echo "pb88: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "pb88: pilot bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "pb88 $TAG: rule=$M pilot=$P ($n_pt members)"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 540000 \
    --bank "$BANK" --method "$M" $EXTRA --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds "$TRAIN" \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/pb88/${TAG}"
echo "pb88 $TAG done"
