#!/bin/bash
# The one Arm A cell that never ran: zoonomia at 300k.
#
# All five seeds died in four seconds on
#     AssertionError: (299999, 300000)
# The zoonomia 300k pool holds 299,999 rows, not 300,000 -- it is the reservoir with 15.4%
# exact duplicates, and deduplication left it one short of the rung. The trainer asserts
# the pool is exactly D, which is the right default: a silently short pool is how a size
# comparison stops being a size comparison.
#
# One sequence in 300,000 is 3 parts per million, far below the ~0.008 seed noise, so the
# honest fix is to train the pool at its real size and say so rather than to pad it or to
# relax the assertion for every reservoir. The output directory records --D 299999, so the
# row carries its own caveat.
#SBATCH --job-name=zoo300k
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/wed41/logs/zoo300k_%A_%a.log
#SBATCH --array=0-2
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/wed41/logs

SEEDS=(10236470 10236471 10236472)
i=$SLURM_ARRAY_TASK_ID
uv run --no-sync python scripts/program_oct22/bench33_legnet.py \
  --D 299999 --cfg e300_wd05 --seeds "${i}=${SEEDS[$i]}" --ctx 20 --save-weights \
  --pool "$PWD/outputs/grid_pools/k562_nested/zoonomia_d300000.npz" \
  --out-root "$PWD/outputs/wed41"
