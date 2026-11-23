#!/bin/bash
# ab41: Arm B. Start from 30k, acquire 30k more, one round or three.
#
# Every arm starts from the SAME labelled 30k -- the reservoir's own nested prefix -- and
# the same round-0 ensemble, which Arm A already trained with weights saved. So the only
# thing that differs between arms is the selection rule, which is the whole point.
#
# Selection is done by albench.acquisition.run_al and training by the study's canonical
# trainer. There is deliberately only ONE training path: the models reported here are the
# same objects Arm A produces, with the same tuned recipe, the same 240-bp input standard
# and the same 18 evaluation sets, so an Arm A point and an Arm B point are comparable.
#
# The controls are arms, not footnotes. `random` is the floor every rule must beat.
# `badge_epistemic_only` and `badge_kmeanspp_only` exist so a BADGE gain can be attributed
# to the fusion rather than to either half -- both BADGE and BatchBALD have failure modes
# that return a full, plausible batch while selecting at random.
#
# Recipe: the per-size table is tabulated at 10k/30k/100k/300k, and the package says to
# interpolate weight decay rather than epochs between them (55k and 80k were measured and
# both prefer 0.1-0.15). 40k-60k therefore trains with e500, which is wd 0.1.
#
# Intermediate rounds score with 3 members and the final bank is 5, so the reported model
# matches Arm A's ensemble size while the rounds stay cheap.
#SBATCH --job-name=ab41
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/acq41/logs/ab41_%A_%a.log
#SBATCH --array=0-11
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq41/logs outputs/acq41/pools

RES=${RES:-motif_shared_core}
# The pool size and the Arm A bank name both differ per reservoir: only the two motif
# reservoirs reach 1M, and `genomic` is the trainer's default so its directory carries no
# reservoir suffix. Resolve both explicitly and fail loudly rather than training on the
# wrong candidate set.
POOL=${POOL:-}
if [ -z "$POOL" ]; then
  for n in 1000000 300000; do
    c="$PWD/outputs/grid_pools/k562_nested/${RES}_d${n}.npz"
    if [ -f "$c" ]; then POOL="$c"; break; fi
  done
fi
if [ "$RES" = "genomic" ]; then
  SEED0_BANK="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
else
  SEED0_BANK="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20_${RES}"
fi
OUTROOT="$PWD/outputs/acq41/${RES}"
[ -f "$POOL" ] || { echo "no candidate pool for $RES"; exit 1; }
n_pt=$(ls "$SEED0_BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "round-0 bank $SEED0_BANK has $n_pt members, need >=3"; exit 1; }
echo "pool=$POOL  bank=$SEED0_BANK ($n_pt members)"

METHODS=(random epistemic badge badge_epistemic_only badge_kmeanspp_only batchbald)
i=$SLURM_ARRAY_TASK_ID
m=${METHODS[$(( i % 6 ))]}
R=$(( i / 6 == 0 ? 1 : 3 ))
ARM="${RES}_${m}_r${R}"
echo "ab41: reservoir=$RES method=$m rounds=$R"

SEEDS3="0=20236470,1=20236471,2=20236472"
SEEDS5="0=20236470,1=20236471,2=20236472,3=20236473,4=20236474"

cur="$POOL"                       # round 0 labelled set = the nested 30k prefix
curbank="$SEED0_BANK"
step=$(( 30000 / R ))

for r in $(seq 1 "$R"); do
  n=$(( 30000 + r * step ))
  next="$PWD/outputs/acq41/pools/${ARM}_r${r}_d${n}.npz"
  # NOT `[ cond ] && x=...`: under `set -e` a false test makes the line return 1 and kills
  # the job, which would have taken out every single-round arm on its first iteration.
  prev_arg=""
  if [ "$r" -gt 1 ]; then
    prev_arg="--current $PWD/outputs/acq41/pools/${ARM}_r$(( r - 1 ))_d$(( 30000 + (r-1)*step )).npz"
  fi

  if [ ! -f "$next" ]; then
    .venv/bin/python -m albench.acquisition.run_al \
      --pool "$POOL" $prev_arg --seed-size 30000 \
      --bank "$curbank" --method "$m" --select "$step" \
      --seed 0 --out "$next"
  fi

  if [ "$r" -eq "$R" ]; then SEEDS="$SEEDS5"; else SEEDS="$SEEDS3"; fi
  uv run --no-sync python scripts/program_oct22/bench33_legnet.py \
    --D "$n" --cfg e500 --seeds "$SEEDS" --ctx 20 --save-weights \
    --pool "$next" --out-root "$OUTROOT/${ARM}_r${r}"
  curbank="$OUTROOT/${ARM}_r${r}/bench38_ln_e500_$(( n / 1000 ))k_c20_${ARM}_r${r}"
  cur="$next"
done
echo "ab41 $ARM done"
