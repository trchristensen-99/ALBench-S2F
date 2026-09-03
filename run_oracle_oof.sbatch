#!/bin/bash
# Out-of-fold oracle evaluation. Stage A (foldmap) first, then this as an array 0-9.
#   sbatch --array=0-9 run_oracle_oof.sbatch
#SBATCH --job-name=orc_oof
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
set -euo pipefail
cd "${ROOT:-/grid/wsbs/home_norepl/christen/ALBench-S2F}"
export TF_CPP_MIN_LOG_LEVEL=3 TQDM_DISABLE=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
uv run --no-sync python experiments/oracle_oof_eval.py --stage predict \
  --fold_id "${SLURM_ARRAY_TASK_ID}"
