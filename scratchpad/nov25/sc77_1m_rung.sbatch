#!/bin/bash
# sc77: the 1,000,000 rung, which is the cheapest way to shrink the decision figure's
# biggest weakness.
#
# fig24 fits a power law on 10k/30k/100k/300k and extends it to a 3.2M order -- a full
# decade past the last measured point. Its own holdout check says the law runs OPTIMISTIC
# out of sample (predicted/actual spans 0.66x-1.09x at 300k when fit on the three smaller
# rungs). Measuring 1M halves that extrapolation: the fit then reaches 3.2M from 1M rather
# than from 300k, and the same holdout check can be rerun one rung further out to say
# whether the optimism grows or settles.
#
# Only the two motif reservoirs have 1M pools, so this cannot be a six-reservoir row and
# the figure must keep saying so. It is still worth having: motif_shared_core is the
# reservoir every objective in the draft order spends most of its budget on, so it is the
# curve whose extrapolation carries the most weight.
#
# One seed per task rather than three per task, so three V100s work on a reservoir at once
# instead of one working three times as long. They share an out-root, which is how the
# Wednesday arms were built and is what cell_dir expects.
#SBATCH --job-name=sc77
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=1-12:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/sc77_%A_%a.log
#SBATCH --array=0-5%6
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/sc77" /tmp/$USER/mpl /tmp/$USER/tmp

RES=(motif_shared_core motif_ct_enriched)
SEEDS=(10236470 10236471 10236472)
i=$SLURM_ARRAY_TASK_ID
r=${RES[$(( i / 3 ))]}
si=$(( i % 3 ))
POOL="$REPO/outputs/grid_pools/k562_nested/${r}_d1000000.npz"

[ -f "$POOL" ] || { echo "sc77: no 1M pool for $r"; exit 1; }
echo "sc77: $r seed $si (e200_wd05, the 1M row of the tuned table)"

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D 1000000 --cfg e200_wd05 --seeds "${si}=${SEEDS[$si]}" \
  --ctx 20 --save-weights --pool "$POOL" --out-root "$KOO/outputs/sc77"
echo "sc77 $r s$si done"
