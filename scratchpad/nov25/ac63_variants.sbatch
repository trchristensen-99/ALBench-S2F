#!/bin/bash
# ac63: "which VARIANT of each strategy does the rule take?" asked where it is cleanly
# measurable -- as a reservoir axis, not as a recovered attribute.
#
# selection_subsets.py can answer this inside a reservoir only where generation
# reproduces byte-identically. It does for motif_shared_core (and the answer is sharp:
# uncertainty takes 7-site plantings at 1.34x and 4-site at 0.85x). It does NOT for
# evoaug or genomic, whose samplers draw from a base pool whose order is not recoverable
# from the stored stamp -- so for those families the honest route is to put the variants
# in the pool as separate reservoirs and let the rule choose between them.
#
#   v30_model   evo2 vs d3_steps64 vs d3_steps256   -- "Evo2 or D3, and at what budget"
#   v30_evoaug  evoaug vs evoaug_ours_default vs evoaug_paper2023
#   v30_motif   motif_shared_core vs motif_ct_enriched vs motif_syntax_core
#
# All three seed from the 30k GENOMIC bank, deliberately: seeding from a member of the
# family under test would make that member's sequences the ones the model has already
# seen, and "the rule avoids what it was trained on" would masquerade as a preference.
#
# random is here as the floor. If it comes back at 0.333/0.333/0.333 the measurement is
# working; if the rule comes back there too, the rule has no preference and that is the
# result.
#SBATCH --job-name=ac63
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ac63_%A_%a.log
#SBATCH --array=0-5%3
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments" ALBENCH_REPO="$REPO"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/pools" "$KOO/outputs/ac63" /tmp/$USER/mpl /tmp/$USER/tmp

FAMS=(v30_model v30_evoaug v30_motif)
MEMBERS=("evo2:1 d3_steps64:1 d3_steps256:1"
         "evoaug:1 evoaug_ours_default:1 evoaug_paper2023:1"
         "motif_shared_core:1 motif_ct_enriched:1 motif_syntax_core:1")
METHODS=(random epistemic)

i=$SLURM_ARRAY_TASK_ID
f=$(( i / 2 ))
m=${METHODS[$(( i % 2 ))]}
FAM=${FAMS[$f]}
TAG="${FAM}_${m}"
POOL="$KOO/pools/${FAM}.npz"
BANK="$REPO/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
SEED=30000; STEP=30000; N=60000

# Built here rather than in a stage job because these three pools are small and the
# builder is atomic: two tasks may both build, wastefully but identically.
if [ ! -f "$POOL" ]; then
  .venv/bin/python "$KOO/bin/build_pool_mix.py" --seed-res genomic --seed-n "$SEED" \
      --members ${MEMBERS[$f]} --cand-n 300000 --out "$POOL"
fi

n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ac63: seed bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ac63 $TAG: ${MEMBERS[$f]}"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 400000 \
    --bank "$BANK" --method "$m" --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds 0=63236470,1=63236471,2=63236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/ac63/${TAG}"
echo "ac63 $TAG done"
