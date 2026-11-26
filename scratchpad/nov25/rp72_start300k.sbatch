#!/bin/bash
# rp72: the setting the collaborators are actually in -- every real CRE already spent.
#
# Seed: genomic at 300,000, which is the whole real-CRE pool. There is no more real
# sequence to acquire, by construction and not by choice. Candidates: the six synthetic
# reservoirs. So this arm asks the question the order actually poses -- "you have all the
# real sequence there is; what should the other 90% of the library be?" -- rather than the
# 30k-start question, which still has real CREs left over and so cannot pose it.
#
# Two selection seeds per rule, because this is the setting that will be quoted and an
# unreplicated number should not be. Three rules rather than six: random as the floor,
# epistemic as the incumbent, coverage_epistemic as the new prior. The remaining rules can
# be added once rp71 says which of them separate at all at 30k.
#SBATCH --job-name=rp72
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/rp72_%A_%a.log
#SBATCH --array=0-5%2
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/rp72" /tmp/$USER/mpl /tmp/$USER/tmp

SIX="random:1 motif_shared_core:1 motif_ct_enriched:1 evoaug_paper2023:1 mutagenesis:1 zoonomia_orthologs:1"
POOL="$KOO/pools/g300_u6.npz"
BANK="$REPO/outputs/wed41/bench38_ln_e300_wd05_300k_c20"   # genomic at 300k, weights kept
SEED=300000; STEP=150000; N=450000

METHODS=(random epistemic coverage_epistemic)
SEEDS=(0 1)
i=$SLURM_ARRAY_TASK_ID
M=${METHODS[$(( i % 3 ))]}
S=${SEEDS[$(( i / 3 ))]}
TAG="g300_u6_${M}_s${S}"

# Built here under the builder's atomic guard: two tasks may race, wastefully but
# identically, and a visible file is always a complete one.
if [ ! -f "$POOL" ]; then
  .venv/bin/python "$KOO/bin/build_pool_mix.py" --seed-res genomic --seed-n "$SEED" \
      --members $SIX --cand-n 700000 --out "$POOL"
fi

n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "rp72: seed bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "rp72 $TAG: method=$M selection-seed=$S  ${SEED}+${STEP} -> ${N}"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 700000 \
    --bank "$BANK" --method "$M" --select "$STEP" --seed "$S" --out "$NEXT"
fi

# e300_wd05 is the 300k row of the tuned table and 450k is one step past it. Between-rule
# differences are safe; the absolute level is one step of extrapolation.
.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e300_wd05 --seeds 0=72236470,1=72236471,2=72236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/rp72/${TAG}"
echo "rp72 $TAG done"
