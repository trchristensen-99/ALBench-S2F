#!/bin/bash
# ye48: AlphaGenome S2 yeast oracle, ONE FOLD, as a decision point (point 6).
#
# Staged deliberately. One fold costs ~6 GPU-h; ten cost sixty, and there is no reason to
# spend sixty before knowing whether the first clears the bar. The bar was fixed before
# the number was seen, in the module docstring: < 0.90 unusable, 0.90-0.955 judge only,
# >= ~0.955 parity and we adopt it as the labeller for consistency with the human task.
#
# Warm-started from the matching S1 fold rather than cold, as agreed -- the AG-FT repo
# trains S2 from scratch, but we already have ten S1 heads and discarding them would make
# this a harder optimisation for no reason.
#SBATCH --job-name=ye48
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=outputs/yeast_ag_s2/logs/ye48_%j.log
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/yeast_ag_s2/logs

uv run --no-sync python experiments/train_ag_yeast_s2_oracle.py \
  --fold-id 0 --out-dir outputs/yeast_ag_s2
