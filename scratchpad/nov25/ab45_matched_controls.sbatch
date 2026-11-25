#!/bin/bash
# ab45: the controls ab44's four rules have to be read against.
#
# ab44 ran activity_normalized, expected_activity and diversity_kmeans on both motif
# reservoirs at cfg e500, D=60k, seed-size 30k, five fixed seeds, and the default
# --cap 200000. The obvious comparison -- Arm B's own epistemic and random at 60k -- is
# NOT available: those arms are cfg e500_wd15, a different recipe, so a difference between
# them and ab44 would be partly recipe. ab43's random arms are cfg e500 but were scored at
# a 970k cap, and ab43 itself shows the cap is worth ~+0.03 on the designed endpoint, so
# they are not matched either.
#
# This runs epistemic and random at EXACTLY ab44's settings, which is the only way the
# four new rules become interpretable. Every field below is copied from
# scratchpad/nov24/ab44_newrules.sbatch unchanged except the method list and the output
# root; the one deliberate difference is the GPU, see below.
#
# V100, not H100. At 04:25 today the cluster had zero free H100s and 28 free V100s, and
# our own slow_nice V100 sub-cap was at 2 of 20 -- the exact inverse of last night, when
# the V100 sub-cap was saturated and the H100 allocation empty. A V100 is ~42% of an H100
# on LegNet, so this is ~3.1 h instead of ~1.3 h, but it starts NOW rather than queueing
# behind a full H100 partition, and the arm is wanted for a meeting this afternoon.
#SBATCH --job-name=ab45
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/acq44/logs/ab45_%A_%a.log
#SBATCH --array=0-3
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq44/logs outputs/acq44/pools

RES=(motif_shared_core motif_ct_enriched)
METHODS=(epistemic random)
i=$SLURM_ARRAY_TASK_ID
r=${RES[$(( i / 2 ))]}
m=${METHODS[$(( i % 2 ))]}

POOL="$PWD/outputs/grid_pools/k562_nested/${r}_d1000000.npz"
BANK="$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20_${r}"
NAME="${r}_${m}"

n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ab45: reservoir=$r method=$m bank=$n_pt members"

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
echo "ab45 $NAME done"
