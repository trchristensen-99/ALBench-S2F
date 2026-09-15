#!/bin/bash
#SBATCH --job-name=sctrain
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:30:00
#SBATCH --array=1-105%20
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/screentrain/%A_%a.out
# Student training for the human 30k parameter screen.
#
# No --constraint=h100 on purpose. LegNet is a CNN and pays only ~2.4x on a V100,
# against ~7.6x for the LSTM oracle, and idle V100s are far easier to get than
# H100s -- constraining this array would make it queue behind the yeast folds for
# a speedup it barely benefits from.
set -uo pipefail
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
CELL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" outputs/screen/train_cells.txt)
[ -z "$CELL" ] && { echo "no cell at index ${SLURM_ARRAY_TASK_ID}"; exit 0; }
OUT=outputs/screen/train/$CELL
# Idempotent: a finished cell is skipped, so the array can be resubmitted freely.
# The driver writes <cell>/n<N>/hp<i>/seed<s>/result.json -- NOT <cell>/results.json.
# Checking the wrong path made the skip silently never fire, which would re-run every
# finished cell on resubmission and leave the watchdog reading 0 trained forever.
if [ -n "$(find "$OUT" -name result.json -size +0c -print -quit 2>/dev/null)" ]; then
  echo "SKIP already done: $CELL"; exit 0
fi
echo "RUN $CELL"
uv run --no-sync python experiments/exp1_1_scaling.py \
  --task k562 --student legnet --oracle ag_s2 \
  --reservoir "$CELL" \
  --pool-base-dir outputs/screen/pools \
  --training-sizes 30000 --n-replicates 1 --no-hp-sweep --chr-split \
  --save-predictions \
  --output-dir "$OUT"
rc=$?
echo "EXIT=$rc"
exit $rc
