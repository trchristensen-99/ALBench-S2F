#!/bin/bash
# Run one AutoResearch cell-round: parallel_gpu_runner with k=4 parallel models.
# Expects configs.json in $CFG_PATH.
#
# Env vars (passed via sbatch --export):
#   CFG_PATH  — path to parallel_gpu_runner configs.json
#   LABEL     — short label for logging

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

: "${CFG_PATH:?CFG_PATH required}"

uv run --no-sync python scripts/preflight/parallel_gpu_runner.py "$CFG_PATH" 4
