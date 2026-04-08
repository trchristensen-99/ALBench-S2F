#!/bin/bash
#SBATCH --job-name=oracle_dream_rnn_yeast
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --array=0-9%4
#SBATCH --output=logs/oracle_dream_rnn_yeast-%A-%a.out
#SBATCH --error=logs/oracle_dream_rnn_yeast-%A-%a.err

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

FOLD_SPLIT_SEED="${FOLD_SPLIT_SEED:-42}"
ORACLE_OUTPUT_ROOT="${ORACLE_OUTPUT_ROOT:-outputs/oracle_dream_rnn_yeast_kfold}"

uv run --no-sync python experiments/train_oracle_dream_rnn.py \
  ++data_path=data/yeast \
  ++n_folds=10 \
  ++fold_id="${SLURM_ARRAY_TASK_ID}" \
  ++fold_split_seed="${FOLD_SPLIT_SEED}" \
  ++seed=null \
  ++output_dir="${ORACLE_OUTPUT_ROOT}/oracle_${SLURM_ARRAY_TASK_ID}" \
  ++wandb_mode=offline
