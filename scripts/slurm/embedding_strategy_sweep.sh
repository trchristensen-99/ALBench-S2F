#!/bin/bash
# Sweep different embedding extraction strategies for Borzoi and Enformer.
# Tests multiple padding, pooling, and context configurations to find optimal.
#
# For each strategy, builds a small embedding cache (val set only for speed),
# trains a head, and reports val Pearson R.
#
# Array:
#   --- Borzoi strategies ---
#   0: Borzoi current (all-bins mean-pool, 200bp flanks, zero-pad)
#   1: Borzoi center-20-bins (only center 20 bins, 200bp flanks)
#   2: Borzoi center-32-bins
#   3: Borzoi center-64-bins
#   4: Borzoi all-bins, 300bp flanks (use full vector)
#   5: Borzoi center-20, 300bp flanks
#   6: Borzoi max-pool (instead of mean)
#   7: Borzoi center-20-bins + max-pool
#   --- Enformer strategies ---
#   8:  Enformer current (center-4-bins, 200bp flanks)
#   9:  Enformer center-6-bins
#   10: Enformer center-8-bins
#   11: Enformer center-12-bins
#   12: Enformer all-896-bins (mean pool)
#   13: Enformer center-4, 300bp flanks
#   14: Enformer center-8, 300bp flanks
#   --- NTv3 strategies ---
#   15: NTv3 current (mean-pool tokens, 200bp flanks)
#   16: NTv3 no flanks (just 200bp insert)
#   17: NTv3 300bp flanks
#
# Usage:
#   sbatch --array=0-17 scripts/slurm/embedding_strategy_sweep.sh
#
#SBATCH --job-name=emb_sweep
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

T=$SLURM_ARRAY_TASK_ID
echo "=== Embedding Strategy Sweep task=${T} ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Strategy definitions
# Format: MODEL FLANK_SIZE CENTER_BINS POOL_MODE
STRATEGIES=(
    "borzoi 200 0 mean"          # 0: current (0 = all bins)
    "borzoi 200 20 mean"         # 1: center-20
    "borzoi 200 32 mean"         # 2: center-32
    "borzoi 200 64 mean"         # 3: center-64
    "borzoi 300 0 mean"          # 4: full vector flanks, all bins
    "borzoi 300 20 mean"         # 5: full vector, center-20
    "borzoi 200 0 max"           # 6: max-pool all bins
    "borzoi 200 20 max"          # 7: center-20 + max-pool
    "enformer 200 4 mean"        # 8: current
    "enformer 200 6 mean"        # 9: center-6
    "enformer 200 8 mean"        # 10: center-8
    "enformer 200 12 mean"       # 11: center-12
    "enformer 200 0 mean"        # 12: all bins
    "enformer 300 4 mean"        # 13: full vector, center-4
    "enformer 300 8 mean"        # 14: full vector, center-8
    "ntv3_post 200 0 mean"       # 15: current
    "ntv3_post 0 0 mean"         # 16: no flanks
    "ntv3_post 300 0 mean"       # 17: full vector flanks
)

read -r MODEL FLANK CENTER_BINS POOL <<< "${STRATEGIES[$T]}"
echo "Model: ${MODEL}, Flank: ${FLANK}bp, Center bins: ${CENTER_BINS} (0=all), Pool: ${POOL}"

OUT_DIR="outputs/embedding_sweep/${MODEL}_f${FLANK}_c${CENTER_BINS}_${POOL}"
mkdir -p "${OUT_DIR}"

# Build cache + train head using a Python script
uv run --no-sync python3 scripts/embedding_strategy_experiment.py \
    --model "${MODEL}" \
    --flank-size "${FLANK}" \
    --center-bins "${CENTER_BINS}" \
    --pool-mode "${POOL}" \
    --output-dir "${OUT_DIR}" \
    --data-path data/k562 \
    --cell-line k562

echo "Done: $(date)"
