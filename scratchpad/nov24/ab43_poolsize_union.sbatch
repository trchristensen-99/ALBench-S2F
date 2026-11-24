#!/bin/bash
# ab43: how big should the candidate pool be, and what if there is more than one?
#
# Two questions Arm B could not answer, both asked on Nov 24.
#
# (1) POOL SIZE. Arm B gave every rule a nominal 1M candidate pool, but run_al caps the
# candidates it SCORES at 200,000 and takes a uniform subsample above that
# (albench/acquisition/run_al.py:111). So every published Arm B arm selected from 200k
# candidates no matter which pool it was pointed at, and the 1M pools were never used as
# 1M. That makes the pool-size axis both unmeasured and the cleanest experiment available:
# vary --cap on ONE pool and the seed set, the reservoir, the recipe and the labelling
# oracle are all held byte-identical, so the only thing that changes is how many
# candidates the rule had in view. Arm B's existing row IS the cap=200k point, so only the
# other rungs need running.
#
# The hypothesis worth stating before the numbers land: epistemic uncertainty selects
# strongly high-activity rows (label mean 2.38 vs a pool mean of 1.21 on shared_core), so a
# larger candidate set lets it go further into the tail. That is a gain if the tail is
# informative and a loss if the tail is just extreme -- a bigger pool cannot help a rule
# whose ranking is already saturated, and it can hurt one that is chasing outliers. A flat
# or falling curve from 50k to 970k is therefore a real and useful result, not a null.
#
# (2) MORE THAN ONE RESERVOIR. motif_shared_core and motif_ct_enriched win DIFFERENT
# endpoints at 60k, which is the case for a mixture, and nobody has asked an acquisition
# rule to choose the mixture itself. The union arm hands it both reservoirs at once from a
# shared 30k seed and the headline is the fraction it takes from each -- a measured mixture
# recommendation rather than a hand-set ratio. Both union arms score the same 970k
# candidates as the largest single-reservoir arm, so the union is not simply being given
# more compute.
#
# `random` runs at the largest cap in both families as the null that makes the rest
# readable: a uniform draw from a nested pool is distributionally the same at any cap, so
# random MUST be flat across the axis. If it is not, the axis is measuring pool
# composition rather than selection and the epistemic rows cannot be read.
#SBATCH --job-name=ab43
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/acq43/logs/ab43_%A_%a.log
#SBATCH --array=0-5
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq43/logs outputs/acq43/pools

SC="$PWD/outputs/grid_pools/k562_nested/motif_shared_core_d1000000.npz"
UNION="$PWD/outputs/acq43/pools/union_sc_ct_d1970000.npz"

# name | pool | method | cap
ARMS=(
  "cap50k_epi|SC|epistemic|50000"
  "cap500k_epi|SC|epistemic|500000"
  "cap970k_epi|SC|epistemic|970000"
  "cap970k_rand|SC|random|970000"
  "union_epi|UNION|epistemic|970000"
  "union_rand|UNION|random|970000"
)
IFS='|' read -r NAME WHICH METHOD CAP <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"

# The union pool is built by whichever union task gets there first; the builder skips an
# existing file, so a second task costs one stat call rather than a duplicate 400MB write.
if [ "$WHICH" = "UNION" ]; then
  .venv/bin/python scratchpad/nov24/build_union_pool.py --out "$UNION"
  POOL="$UNION"
else
  POOL="$SC"
fi
[ -f "$POOL" ] || { echo "no candidate pool at $POOL"; exit 1; }

# Round-0 bank is Arm A's own 30k shared_core bank -- the same one every Arm B arm scored
# with, so the selection is made by an identical model and only the candidate set differs.
BANK="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20_motif_shared_core"
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "round-0 bank $BANK has $n_pt members, need >=3"; exit 1; }

echo "ab43 $NAME: pool=$(basename "$POOL") method=$METHOD cap=$CAP bank=$n_pt members"

NEXT="$PWD/outputs/acq43/pools/${NAME}_d60000.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size 30000 \
    --bank "$BANK" --method "$METHOD" --select 30000 \
    --cap "$CAP" --seed 0 --out "$NEXT"
fi

# e500 and these five seeds are Arm B's own settings at 60k, so these rows drop straight
# into the Arm B table instead of forming a separate one.
uv run --no-sync python scripts/program_oct22/bench33_legnet.py \
  --D 60000 --cfg e500 \
  --seeds 0=20236470,1=20236471,2=20236472,3=20236473,4=20236474 \
  --ctx 20 --save-weights \
  --pool "$NEXT" --out-root "$PWD/outputs/acq43/${NAME}"
echo "ab43 $NAME done"
