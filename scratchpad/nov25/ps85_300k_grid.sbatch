#!/bin/bash
# ps85: the pool-size x rule grid again, at the 300k start instead of the 30k one.
#
# ps68/ps73 measure how pool size interacts with the rule when the model has seen 30,000
# real CREs. The order is placed from 300,000, and there is no reason the interaction
# transfers: at 30k the model is data-starved and almost any informative batch helps, while
# at 300k it already fits the real-CRE distribution well and the question becomes which
# synthetic sequence adds something it does not have. A pool-size curve measured only at
# 30k would be quoted for a decision made at 300k.
#
# Three candidate-pool sizes crossed with random and uncertainty, from the genomic 300,000
# seed -- every real CRE already spent -- choosing from the six synthetic reservoirs.
#
# WHY 1.5M IS THE TOP RUNG AND NOT MORE. Four of the six reservoirs hold 300,000 sequences,
# so an even six-way split caps at 1.8M before any of them runs short, and zoonomia_orthologs
# loses 15.4% of its rows to deduplication on top of that. 1.5M (250,000 each) is the
# largest rung every member can actually fill, and the builder reports any shortfall rather
# than topping up from elsewhere.
#SBATCH --job-name=ps85
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ps85_%A_%a.log
#SBATCH --array=0-5%6
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/ps85" "$KOO/pools" /tmp/$USER/mpl /tmp/$USER/tmp

SIX="random:1 motif_shared_core:1 motif_ct_enriched:1 evoaug_paper2023:1 mutagenesis:1 zoonomia_orthologs:1"
SIZES=(200000 700000 1500000)
METHODS=(random epistemic)
BANK="$REPO/outputs/wed41/bench38_ln_e300_wd05_300k_c20"
SEED=300000; STEP=150000; N=450000

i=$SLURM_ARRAY_TASK_ID
SZ=${SIZES[$(( i / 2 ))]}
M=${METHODS[$(( i % 2 ))]}
POOL="$KOO/pools/g300_u6_${SZ}.npz"
TAG="g300_u6_${SZ}_${M}"

# Built under the builder's atomic guard; the two rules at a size may race and will write
# identical content, or one will see the other's finished file.
if [ ! -f "$POOL" ]; then
  .venv/bin/python "$KOO/bin/build_pool_mix.py" --seed-res genomic --seed-n "$SEED" \
      --members $SIX --cand-n "$SZ" --out "$POOL"
fi
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ps85: bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ps85 $TAG: candidates=$SZ method=$M  ${SEED}+${STEP} -> ${N}"

# Score the whole pool: pool size is the variable, so a cap below it would flatten the axis.
NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap "$SZ" \
    --bank "$BANK" --method "$M" --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e300_wd05 --seeds 0=85236470,1=85236471,2=85236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/ps85/${TAG}"
echo "ps85 $TAG done"
