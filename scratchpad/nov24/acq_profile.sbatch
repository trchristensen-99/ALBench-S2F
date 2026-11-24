#!/bin/bash
# What activity does each acquisition rule actually buy? (point 5, Nov 24)
#
# No training: this scores the real candidate pool with Arm A's own bank and reports what
# each rule selects. The question is whether `epistemic` is selecting uncertainty or
# selecting activity -- activities are log2FC and absolute error grows with predicted
# magnitude, so raw ensemble spread partly re-ranks by activity. Arm B is consistent with
# that (epistemic bought a label mean of 2.38 against a pool mean of 1.21).
#SBATCH --job-name=acqprof
#SBATCH --partition=gpuq
#SBATCH --account=cshlsbs
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/acq43/logs/acqprof_%A_%a.log
#SBATCH --array=0-1
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/experiments"
mkdir -p outputs/acq43/logs

RES=(motif_shared_core motif_ct_enriched)
r=${RES[$SLURM_ARRAY_TASK_ID]}

uv run --no-sync python scratchpad/nov24/acq_activity_profile.py \
  --pool "$PWD/outputs/grid_pools/k562_nested/${r}_d1000000.npz" \
  --bank "$PWD/outputs/wed41/bench38_ln_e500_wd2_30k_c20_${r}" \
  --out "$PWD/outputs/acq43/activity_profile_${r}.json"
