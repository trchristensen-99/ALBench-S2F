#!/bin/bash
# pm74 stage 2: train the eight PRM-ladder corpora.
#
# All at 30,000 with the tuned 30k recipe, so the only thing varying across the eight is
# how the mutations were applied. The existing `mutagenesis` cell at 30k is the reference
# point already in every figure, and prm_once_k1 should land on it -- one pass at 0.05 is
# what that reservoir is. If it does not, the ladder is measuring something other than
# what it claims and nothing else in the family should be read.
#SBATCH --job-name=pm74
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/grid/koo/home/christen/nov25_extra/logs/pm74_%A_%a.log
#SBATCH --array=0-7%8
set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
KOO=/grid/koo/home/christen/nov25_extra
cd "$REPO"
export PYTHONPATH="$PWD:$PWD/experiments"
export MPLCONFIGDIR=/tmp/$USER/mpl TMPDIR=/tmp/$USER/tmp HP_ALLOW_UNSTAMPED=1
mkdir -p "$KOO/logs" "$KOO/outputs/pm74" /tmp/$USER/mpl /tmp/$USER/tmp

mapfile -t POOLS < <(ls "$KOO"/prm_ladder/*_d30000.npz | sort)
P=${POOLS[$SLURM_ARRAY_TASK_ID]:-}
[ -n "$P" ] || { echo "pm74: no pool at index $SLURM_ARRAY_TASK_ID"; exit 1; }
TAG=$(basename "$P" _d30000.npz)
echo "pm74 $TAG: $(basename "$P")"

.venv/bin/python scripts/program_oct22/bench33_legnet.py \
  --D 30000 --cfg e500_wd2 --seeds 0=74236470,1=74236471,2=74236472 \
  --ctx 20 --save-weights --pool "$P" --out-root "$KOO/outputs/pm74/${TAG}"
echo "pm74 $TAG done"
