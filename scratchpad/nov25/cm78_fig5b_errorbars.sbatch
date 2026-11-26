#!/bin/bash
# cm78: error bars on the one claim fig5b actually makes.
#
# fig5b's headline is that no mixture beats the best single reservoir, across 35 arms and
# three rules. Every one of those arms used selection seed 0, so the claim currently rests
# on a single draw and has no error bar -- which is exactly the kind of statement that
# should not go into a registered report unreplicated.
#
# Rather than replicate all 35, this replicates the two compositions the claim is ABOUT:
# motif_shared_core alone (the best single) and the four-way mix (the best mixture), under
# all three rules, at two further selection seeds. If the gap between them is smaller than
# the spread across seeds, "no mixture beats the best single" becomes "we cannot tell",
# which is a different and more honest sentence.
#
# Same construction as ab53: a 300,000-row candidate pool split evenly across whichever
# reservoirs the composition names, so composition varies and the amount to choose from
# does not.
#SBATCH --job-name=cm78
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=default
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/cm78_%A_%a.log
#SBATCH --array=0-11%4
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/cm78" "$KOO/pools" /tmp/$USER/mpl /tmp/$USER/tmp

# index: composition (2) x rule (3) x selection seed (2)
COMPS=("motif_shared_core=300000" "random=75000 motif_shared_core=75000 evoaug=75000 genomic=75000")
CNAME=(single4x mix4way)
METHODS=(uncertainty random expected_activity)
SEEDS=(1 2)
D=30000
BANK="$REPO/outputs/wed41/bench38_ln_e500_wd2_30k_c20_motif_shared_core"

i=$SLURM_ARRAY_TASK_ID
c=$(( i % 2 )); m=$(( (i / 2) % 3 )); s=$(( i / 6 ))
TAG="${CNAME[$c]}_${METHODS[$m]}_s${SEEDS[$s]}"
POOL="$KOO/pools/cm78_${TAG}_d${D}.npz"

n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "cm78: bank $BANK has $n_pt members, need >=3"; exit 1; }

SRC=(); for t in ${COMPS[$c]}; do SRC+=(--source "$t"); done
echo "cm78 $TAG: ${COMPS[$c]} | method=${METHODS[$m]} | selection seed ${SEEDS[$s]}"

if [ ! -f "$POOL" ]; then
  # --cap 300000, not the 200,000 default: the pool IS 300,000 and the original arms
  # scored only two thirds of it. Matching the original would reproduce its blind spot;
  # scoring all of it makes these replicates slightly different from the seed-0 arms, and
  # that difference is stated rather than hidden.
  .venv/bin/python scratchpad/nov25/select_corpus.py \
    --out "$POOL" --select "$D" --bank "$BANK" --method "${METHODS[$m]}" \
    --cap 300000 --seed "${SEEDS[$s]}" "${SRC[@]}"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$D" --cfg e500_wd2 --seeds 0=78236470,1=78236471,2=78236472 \
  --ctx 20 --save-weights --pool "$POOL" --out-root "$KOO/outputs/cm78/${TAG}"
echo "cm78 $TAG done"
