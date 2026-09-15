#!/bin/bash
#SBATCH --job-name=yoracleB
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/yoracleB/%A_%a.out
set -uo pipefail
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
cd "${ALBENCH_REPO:-/grid/wsbs/home_norepl/christen/ALBench-S2F}"
uv run --no-sync python experiments/train_yeast_oracle_v2.py \
  --fold-id "${SLURM_ARRAY_TASK_ID}" --epochs 6 --num-workers 8 --resume --tag _B --include-eval-classes snv,challenging,native_genomic,high_expression,low_expression \
  --out-dir outputs/yeast_oracle_v2
rc=$?
[ $rc -eq 0 ] && echo "OK fold ${SLURM_ARRAY_TASK_ID}" || { echo "FAIL rc=$rc fold ${SLURM_ARRAY_TASK_ID}"; exit 1; }
