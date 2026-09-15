#!/bin/bash
#SBATCH --job-name=yopred
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/yopredict/%A_%a.out
# Score the 71,103 DREAM eval sequences with every trained yeast-oracle fold.
# Tasks 0-9 are variant A, 10-19 are variant B, so both ensembles are covered by one
# array. h100-constrained: this is the same LSTM that pays 7.6x on a V100.
set -uo pipefail
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
cd "${ALBENCH_REPO:-/grid/wsbs/home_norepl/christen/ALBench-S2F}"
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 10 ]; then FOLD=$i; TAG=""; else FOLD=$((i-10)); TAG="_B"; fi
OUT="outputs/yeast_oracle_v2/fold_${FOLD}${TAG}/eval_class_predictions.npz"
if [ -s "$OUT" ]; then echo "SKIP already present: $OUT"; exit 0; fi
echo "RUN fold=$FOLD tag="
uv run --no-sync python scripts/predict_yeast_eval_classes.py --fold-id "$FOLD" --tag "$TAG"
rc=$?; echo "EXIT=$rc"; exit $rc
