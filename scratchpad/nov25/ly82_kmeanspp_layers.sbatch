#!/bin/bash
# ly82: k-means++ at each layer, which is the rule the layer question was asked about.
#
# ly81 runs `diversity_kmeans`, which despite its name is greedy farthest-point (LCMD).
# The rule in the acquisition figures is `badge_kmeanspp_only` -- actual k-means++ seeding.
# Both measure distance through embed(), so both inherit the layer, but they behave very
# differently in a degenerate space: farthest-point is deterministic and takes extremes, so
# in a 1-D activity-aligned embedding it selects the top and bottom of the activity range
# and little else, while k-means++ samples proportional to squared distance and spreads
# more evenly. Running only one of them would confound "which layer" with "which rule".
#
# Same pool, same seed, same five layers as ly81, so the two families form a 2 x 5 grid.
#SBATCH --job-name=ly82
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/ly82_%A_%a.log
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
TAG="u6_540k_kmeanspp_L${L//./_}"

[ -f "$POOL" ] || { echo "ly82: candidate pool missing: $POOL"; exit 1; }
n_pt=$(ls "$BANK"/*.pt 2>/dev/null | wc -l)
[ "$n_pt" -ge 3 ] || { echo "ly82: bank $BANK has $n_pt members, need >=3"; exit 1; }
echo "ly82 $TAG: badge_kmeanspp_only at layer $L"

NEXT="$KOO/pools/sel_${TAG}_d${N}.npz"
if [ ! -f "$NEXT" ]; then
  .venv/bin/python -m albench.acquisition.run_al \
    --pool "$POOL" --seed-size "$SEED" --cap 200000 \
    --bank "$BANK" --method badge_kmeanspp_only --embed-layer "$L" \
    --select "$STEP" --seed 0 --out "$NEXT"
fi

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$N" --cfg e500_wd2 --seeds 0=81236470,1=81236471,2=81236472 \
  --ctx 20 --save-weights --pool "$NEXT" --out-root "$KOO/outputs/ly81/${TAG}"
echo "ly82 $TAG done"
