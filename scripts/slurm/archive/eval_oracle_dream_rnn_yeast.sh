#!/bin/bash
# Eval-only: run test evaluation on trained DREAM-RNN oracle folds.
# For folds that have best_model.pt but no summary.json (timed out before eval).
#
# Submit for folds 4-7:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=4-7 scripts/slurm/eval_oracle_dream_rnn_yeast.sh
#
# Environment overrides:
#   ORACLE_OUTPUT_ROOT  — oracle checkpoint dir (default: outputs/oracle_dream_rnn_yeast_kfold_v256_rcaug)
#   FOLD_SPLIT_SEED     — k-fold split seed (default: 42)
#
#SBATCH --job-name=eval_oracle_yeast
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "Job ID: $SLURM_JOB_ID"
echo "Array task (fold): $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

ORACLE_OUTPUT_ROOT="${ORACLE_OUTPUT_ROOT:-outputs/oracle_dream_rnn_yeast_kfold_v256_rcaug}"
FOLD_SPLIT_SEED="${FOLD_SPLIT_SEED:-42}"

uv run --no-sync python scripts/eval_oracle_dream_rnn_yeast.py \
    --oracle-dir "${ORACLE_OUTPUT_ROOT}" \
    --fold-id "${SLURM_ARRAY_TASK_ID}" \
    --fold-split-seed "${FOLD_SPLIT_SEED}" \
    --hidden-dim 320 \
    --cnn-filters 256 \
    --use-reverse-complement

echo "Done at $(date)"
