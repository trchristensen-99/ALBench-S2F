#!/bin/bash
# rp80: more of the 300k start, queued on the `default` tier's unused H100 allocation.
#
# Two reasons this exists. Scientifically, rp72 runs three rules at two selection seeds at
# the 300k start, and that start is the one the registered report will quote -- it is the
# setting where every real CRE is already spent. coverage_random belongs there because it
# is the no-model recipe, and expected_activity belongs there because it won a different
# endpoint than uncertainty at 30k and nobody has checked whether that survives to 300k.
#
# Operationally, the H100 partition is at 72 of 76 cluster-wide, so there is nothing to
# grab right now -- but the `default` QOS allocation (4 H100) is completely unused, and a
# queued job takes a card the instant one frees. Submitting into an empty tier costs
# nothing while the cluster is full and collects capacity the moment it is not.
#SBATCH --job-name=rp80
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/rp80_%A_%a.log
#SBATCH --array=0-3%4
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/rp72" /tmp/$USER/mpl /tmp/$USER/tmp

SIX="random:1 motif_shared_core:1 motif_ct_enriched:1 evoaug_paper2023:1 mutagenesis:1 zoonomia_orthologs:1"
POOL="$KOO/pools/g300_u6.npz"
BANK="$REPO/outputs/wed41/bench38_ln_e300_wd05_300k_c20"
SEED=300000; STEP=150000; N=450000

METHODS=(coverage_random expected_activity)
SEEDS=(0 1)
i=$SLURM_ARRAY_TASK_ID
M=${METHODS[$(( i % 2 ))]}
S=${SEEDS[$(( i / 2 ))]}
TAG="g300_u6_${M}_s${S}"

# rp72 builds the same pool under the same atomic guard; whichever job gets there first
# writes it and the other skips. A visible file is always a complete one.
if [ ! -f "$POOL" ]; then
  .venv/bin/python "$KOO/bin/build_pool_mix.py" --seed-res genomic --seed-n "$SEED" \
      --members $SIX --cand-n 700000 --out "$POOL"
fi
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "rp80: seed bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "rp80 $TAG: method=$M selection-seed=$S  ${SEED}+${STEP} -> ${N}"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 700000 \
    --bank "$BANK" --method "$M" --select "$STEP" --seed "$S" --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e300_wd05 --seeds 0=72236470,1=72236471,2=72236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/rp72/${TAG}"
echo "rp80 $TAG done"
