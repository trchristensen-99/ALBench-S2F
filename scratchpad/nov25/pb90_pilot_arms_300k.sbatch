#!/bin/bash
# pb90: pilot-replicated arms at the 300k start -- the setting the order is placed from.
#
# pb88 does this at 30k. This is the same design where it matters: seed is genomic at
# 300,000, which is every real CRE in existence, and the candidates are the six synthetic
# reservoirs. Three independent pilots (pb89), three rules, independent training seeds per
# replicate.
#
# Three rules rather than six. At 450,000 rows a cell costs roughly four times a 60k one,
# and rp72/rp80/rp86 already ran all six rules here at a single pilot -- what is missing is
# an error bar, not more rules. random is the floor, epistemic the incumbent,
# coverage_epistemic the prior that was proposed; those three carry the comparison that
# would be quoted.
#SBATCH --job-name=pb90
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/pb90_%A_%a.log
#SBATCH --array=0-8%9
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/pb90" /tmp/$USER/mpl /tmp/$USER/tmp

METHODS=(random epistemic coverage_epistemic)
PILOTS=(A B C)
TRAIN_A="0=90100001,1=90100002,2=90100003"
TRAIN_B="0=90200001,1=90200002,2=90200003"
TRAIN_C="0=90300001,1=90300002,2=90300003"

POOL="$KOO/pools/g300_u6.npz"
SEED=300000; STEP=150000; N=450000

i=$SLURM_ARRAY_TASK_ID
M=${METHODS[$(( i % 3 ))]}
P=${PILOTS[$(( i / 3 ))]}
case $P in
  A) TRAIN=$TRAIN_A ;;
  B) TRAIN=$TRAIN_B ;;
  C) TRAIN=$TRAIN_C ;;
esac
BANK="$KOO/outputs/pb89/pilot${P}/bench38_ln_e300_wd05_300k_c20"
TAG="g300_u6_${M}_pilot${P}"

[ -f "$POOL" ] || { echo "pb90: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "pb90: pilot bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "pb90 $TAG: rule=$M pilot=$P ($n_pt members)  ${SEED}+${STEP} -> ${N}"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 700000 \
    --bank "$BANK" --method "$M" --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e300_wd05 --seeds "$TRAIN" \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/pb90/${TAG}"
echo "pb90 $TAG done"
