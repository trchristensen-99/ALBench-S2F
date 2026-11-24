#!/bin/bash
# ab42: Arm B at the 300k start. The coverage gap, and the one that decides the library.
#
# Arm B measured acquisition at 30k -> 60k and found epistemic +0.0090 genomic / +0.0855
# OOD over random, with the gain attributable to the epistemic half rather than to
# BADGE's diversity term. Nothing says that survives at scale, and scale is the regime
# the collaborators are actually in: a 4M order is not a 60k order, and a rule that helps
# when the model is data-starved can be worth nothing once it is not.
#
# WHY THESE TWO RESERVOIRS AND NOT genomic. The real-CRE pool holds 314,981 distinct
# sequences, so a 300k seed set exhausts it and there is nothing left to acquire -- a
# 300k->450k genomic arm cannot be run at all, which is itself most of the argument for
# why a 4M library cannot be mostly real sequence. motif_shared_core and
# motif_ct_enriched are the only reservoirs with 1M pools, and they are also the two that
# led Arm A (0.8771 and 0.8817 genomic, and motif_shared_core best OOD at 0.5933), so the
# large-budget question gets asked on the reservoirs a library would actually use.
#
# WHY +150k AND NOT +300k. A doubling would cost 87 GPU-h against this arm's 65 and the
# per-size recipe is only tabulated to 300k, so 600k would extrapolate the schedule two
# steps past where it was measured. 450k interpolates one step and is still a realistic
# decision -- "we have 300k, we can afford 150k more".
#
# THREE METHODS, NOT SIX. Arm B already ruled three out with controls: badge is below
# badge_epistemic_only, badge_kmeanspp_only is below badge, and batchbald reported
# n_informative_ = 0 with an acquired label mean of 0.470 against a pool mean of 0.469 --
# it was selecting at random and scored accordingly. Re-running them here would spend
# half the budget confirming a settled result. `random` stays, because it is the floor,
# and badge_epistemic_only stays as the control that attributes any epistemic gain.
#
# RECIPE CAVEAT, to state with the numbers: 450k trains with e300_wd05, the 300k row of
# the tuned table. The package says to interpolate weight decay rather than epochs, and
# 450k is past the last measured row, so this is one step of extrapolation and an
# e300_wd05-vs-e300 pair at 450k would be worth running before anyone leans on the
# absolute level rather than the between-method differences.
#SBATCH --job-name=ab42
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/acq42/logs/ab42_%A_%a.log
#SBATCH --array=0-5
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq42/logs outputs/acq42/pools

RESERVOIRS=(motif_shared_core motif_ct_enriched)
METHODS=(random epistemic badge_epistemic_only)

i=$SLURM_ARRAY_TASK_ID
RES=${RESERVOIRS[$(( i / 3 ))]}
m=${METHODS[$(( i % 3 ))]}

SEED=300000
STEP=150000
N=$(( SEED + STEP ))
POOL="$PWD/outputs/grid_pools/k562_nested/${RES}_d1000000.npz"
SEED0_BANK="$PWD/outputs/wed41/bench38_ln_e300_wd05_300k_c20_${RES}"
ARM="${RES}_${m}_300k"
OUTROOT="$PWD/outputs/acq42/${RES}"

[ -f "$POOL" ] || { echo "no 1M candidate pool for $RES at $POOL"; exit 1; }
n_pt=$(ls "$SEED0_BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "round-0 bank $SEED0_BANK has $n_pt members, need >=3"; exit 1; }
echo "ab42: reservoir=$RES method=$m  ${SEED} + ${STEP} -> ${N}"
echo "      pool=$POOL  bank=$SEED0_BANK ($n_pt members)"

NEXT="$PWD/outputs/acq42/pools/${ARM}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" \
    --bank "$SEED0_BANK" --method "$m" --select "$STEP" \
    --seed 0 --out "$NEXT"
fi

uv run --no-sync python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e300_wd05 --seeds 0=30236470,1=30236471,2=30236472 \
  --ctx 20 --save-weights \
  --pool "$NEXT" --out-root "$OUTROOT/${ARM}"
echo "ab42 $ARM done"
