#!/bin/bash
# cs62: do COMBINATIONS of reservoirs scale, and do they beat what acquisition selects?
#
# fig5b compared 35 mixtures at one size and found no mixture beat the best single
# reservoir. That is a statement about 30k. The order is 4M, and "no mixture helps" and
# "no mixture helps yet" are different recommendations -- a mixture whose value is
# coverage should pay off later, not earlier, because a single generator runs out of new
# things to say before several do.
#
# Two combinations, three sizes, from scratch:
#   combo_u7_flat          the equal 7-way mix -- the same composition the acquisition
#                          arms select FROM, so acq-at-60k reads directly against it
#   combo_ms_evoaug_flat   motif_shared_core + evoaug_paper2023 -- the pair that won
#                          different endpoints in fig19 (motif the designed set, evoaug
#                          the genomic one), which is the only principled reason to mix
#
# The corpora are interleaved by share, so the 30k prefix and the 300k prefix hold the
# same mixture and the size axis varies the amount of data and nothing else. Without that
# the curve would be confounded exactly the way k562_cumulative was.
#SBATCH --job-name=cs62
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/cs62_%A_%a.log
#SBATCH --array=0-5%3
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/cs62" /tmp/$USER/mpl /tmp/$USER/tmp

COMBOS=(combo_u7_flat combo_ms_evoaug_flat)
SIZES=(30000 100000 300000)
CFGS=(e500_wd2 e500 e300_wd05)

i=$SLURM_ARRAY_TASK_ID
c=${COMBOS[$(( i % 2 ))]}
k=$(( i / 2 ))
D=${SIZES[$k]}; CFG=${CFGS[$k]}
POOL="$KOO/pools/${c}_d${D}.npz"

[ -f "$POOL" ] || { echo "cs62: pool missing $POOL"; exit 1; }
echo "cs62: $c D=$D cfg=$CFG"
.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D "$D" --cfg "$CFG" --seeds 0=62236470,1=62236471,2=62236472 \
  --ctx 20 --save-weights --pool "$POOL" --out-root "$KOO/outputs/cs62/${c}"
echo "cs62 $c d$D done"
