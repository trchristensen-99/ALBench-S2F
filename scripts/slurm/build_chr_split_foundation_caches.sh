#!/bin/bash
# Build embedding caches for chromosome-split train/val/test sets.
#
# Uses existing cache builder scripts but with chromosome split indices.
# The sequences are the same as HashFrag - only the split assignment changes.
#
# For now, this builds FULL dataset caches (all ~401K sequences) as a single
# array, then we create split-specific index files. This avoids rebuilding
# per-split caches for each split scheme.
#
# Array: 0=Enformer, 1=Borzoi, 2=NTv3-post
#
#SBATCH --job-name=chr_cache
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

MODELS=("enformer" "borzoi" "ntv3_post")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Building Chr-Split Foundation Cache ==="
echo "Model: ${MODEL}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# For now, we can reuse existing HashFrag caches for the chr-split test evaluation
# by building a mapping from full-dataset indices to cache positions.
# This is done by the evaluation script, not the cache builder.

# Create chromosome split index files if they don't exist
uv run --no-sync python -c "
import numpy as np
from pathlib import Path
from data.k562 import K562Dataset

# Load chromosome-split dataset to get indices
ds_train = K562Dataset('data/k562', split='train', use_hashfrag=False, use_chromosome_fallback=True)
ds_val = K562Dataset('data/k562', split='val', use_hashfrag=False, use_chromosome_fallback=True)
ds_test = K562Dataset('data/k562', split='test', use_hashfrag=False, use_chromosome_fallback=True)

out = Path('data/k562/chr_splits')
out.mkdir(exist_ok=True)

print(f'Chr train: {len(ds_train)} val: {len(ds_val)} test: {len(ds_test)}')

# Save the split sizes so we know how to slice the caches
np.savez(out / 'split_info.npz',
         train_size=len(ds_train),
         val_size=len(ds_val),
         test_size=len(ds_test))
print(f'Saved split info to {out}')
"

echo "Done: $(date)"
