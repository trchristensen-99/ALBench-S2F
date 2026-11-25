#!/bin/bash
# ab44: does epistemic's advantage survive removing its activity shift?
#
# Measured today on the real candidate pool (outputs/acq43/activity_profile_*.json), with
# motif_shared_core's pool label mean at 1.210:
#
#   rule                  label mean   mean uncertainty   frac > pool mean
#   random                   1.213           0.159              0.38
#   epistemic                2.377           0.313              0.76
#   activity_normalized      1.180           0.268              0.38
#   expected_activity        3.843           0.224              1.00
#
# Two things follow. Epistemic is NOT simply ranking by activity -- if it were it would
# reach expected_activity's 3.84 -- but it is substantially activity-shifted, buying
# roughly double the pool's mean activity. And activity_normalized decouples the two: it
# keeps 86% of epistemic's uncertainty gain over random while its label distribution is
# indistinguishable from random's. So "is epistemic's gain an uncertainty gain or an
# activity gain?" is now a question with a clean experiment, which is this one.
#
# expected_activity is in the arm as a serious candidate, not a control: a library meant
# to design strong enhancers has a real reason to spend at the top of the range, and
# nobody has measured what a pure-exploitation corpus does to a student.
#
# diversity_kmeans completes the family set for the registered report -- uncertainty,
# diversity, uncertainty+diversity, expected activity -- so each family has a
# representative that has actually been run end to end.
#
# e500 and these five seeds are Arm B's own settings at 60k, so these rows drop straight
# into the Arm B table rather than forming a separate one.
#SBATCH --job-name=ab44
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/acq44/logs/ab44_%A_%a.log
#SBATCH --array=0-5
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq44/logs outputs/acq44/pools

RES=(motif_shared_core motif_ct_enriched)
METHODS=(activity_normalized expected_activity diversity_kmeans)
i=$SLURM_ARRAY_TASK_ID
r=${RES[$(( i / 3 ))]}
m=${METHODS[$(( i % 3 ))]}

POOL="$PWD/outputs/grid_pools/k562_nested/${r}_d1000000.npz"
BANK="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20_${r}"
NAME="${r}_${m}"

n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ab44: reservoir=$r method=$m bank=$n_pt members"

NEXT="$PWD/outputs/acq44/pools/${NAME}_d60000.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size 30000 \
    --bank "$BANK" --method "$m" --select 30000 \
    --seed 0 --out "$NEXT"
fi

uv run --no-sync python scripts/program_oct22/bench33_legnet.py \
  --D 60000 --cfg e500 \
  --seeds 0=20236470,1=20236471,2=20236472,3=20236473,4=20236474 \
  --ctx 20 --save-weights \
  --pool "$NEXT" --out-root "$PWD/outputs/acq44/${NAME}"
echo "ab44 $NAME done"
