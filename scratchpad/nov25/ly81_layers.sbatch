#!/bin/bash
# ly81: run the diversity rule in each layer's geometry, not just the penultimate one.
#
# WHY. Probed on the 30k bank over 3,000 candidates, the penultimate layer -- the one every
# diversity arm in this study has used -- recovers the model's own prediction with
# R^2 = 1.000, has a first principal component correlating 0.998 with that prediction, and
# carries 1.13 effective dimensions. Running a distance-based rule there is ranking by
# predicted activity with extra steps. The collapse starts at main.3 (R^2 0.998, 1.67
# effective dims); everything after it is the same single direction.
#
# The convolutional blocks are where the geometry is:
#   stem    R^2 0.269, eff_dim  2.84   low-level composition, but only 64 channels
#   main.0  R^2 0.362, eff_dim 25.39   richest and least activity-aligned
#   main.1  R^2 0.699, eff_dim 21.30   a middle: still 21 directions, half-explained
#   main.2  R^2 0.843, eff_dim 10.28   starting to collapse
#   head.2  R^2 1.000, eff_dim  1.13   the current default, i.e. the control
#
# Selections from different layers really do differ -- head.2 against main.0 overlaps 0.16
# where two independent draws would overlap 0.10 -- so this is a choice with consequences
# rather than a tuning detail. head.2 is included precisely so the comparison has the
# incumbent in it, and so that "deeper is worse" is measured rather than assumed from the
# geometry alone.
#
# One selection seed for now: the question here is which layer, and five layers at three
# seeds would spend the night on error bars for an axis nobody has looked at yet.
#SBATCH --job-name=ly81
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ly81_%A_%a.log
#SBATCH --array=0-4%5
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/ly81" /tmp/$USER/mpl /tmp/$USER/tmp

LAYERS=(stem main.0 main.1 main.2 head.2)
BANK="$REPO/outputs/wed41/bench38_ln_e500_wd2_30k_c20"
POOL="$KOO/pools/u6_540000.npz"
SEED=30000; STEP=30000; N=60000

i=$SLURM_ARRAY_TASK_ID
L=${LAYERS[$i]}
TAG="u6_540k_diversity_L${L//./_}"

[ -f "$POOL" ] || { echo "ly81: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ly81: bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ly81 $TAG: diversity_kmeans measured at layer $L"

# --cap 200000 here, not the full pool: the farthest-point rule holds an (N, C) embedding
# and a running distance vector, and main.0 is 80 channels over 540,000 candidates before
# the O(N*k) sweep. The cap is identical across layers, so the layer comparison is fair
# even though it is not the whole pool.
NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 200000 \
    --bank "$BANK" --method diversity_kmeans --embed-layer "$L" \
    --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds 0=81236470,1=81236471,2=81236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/ly81/${TAG}"
echo "ly81 $TAG done"
