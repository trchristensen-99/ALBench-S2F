#!/bin/bash
# dm47: the design-method budget ladder (point 3, Nov 24).
#
# The canary compared the three methods at wildly unequal budgets -- FastSeqProp updates
# all 200 positions every one of 150 steps, while SA got 60 single-base proposals -- so
# "SA and AdaLead reach lower activity" was a statement about budget, not method. Gosai ran
# annealing for thousands of steps. This runs all three along a budget ladder and scores
# every configuration with the independent oracle.
#SBATCH --job-name=dm47
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/design47/logs/dm47_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/design47/logs

uv run --no-sync python scratchpad/nov24/dm47_design_budget.py \
  --n 512 --out "$PWD/outputs/design47/ladder.json"
