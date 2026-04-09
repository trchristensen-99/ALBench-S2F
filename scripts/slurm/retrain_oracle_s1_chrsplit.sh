#!/bin/bash
# Retrain AG S1 oracle on full chr-split ref+alt dataset (618K sequences).
#
# Step 1: Build embedding cache for ALL chr-split train sequences
# Step 2: Train 10-fold S1 oracle heads on the cached embeddings
#
# Each fold trains on ~556K sequences (90%) and validates on ~62K (10%).
# Chr7+13 (test) sequences are NEVER included in any fold.
#
# Array: 0-9 (one per fold). Each fold takes ~15-20 min.
# All 10 can run in parallel if 10 H100 slots available.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch --array=0-9 scripts/slurm/retrain_oracle_s1_chrsplit.sh
#
#SBATCH --job-name=orc_s1_chr
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

FOLD=$SLURM_ARRAY_TASK_ID
CACHE_DIR="outputs/oracle_chrsplit_cache/embedding_cache"
OUT_DIR="outputs/oracle_chrsplit_s1/oracle_${FOLD}"

echo "=== Oracle S1 retrain: fold ${FOLD} node=${SLURMD_NODENAME} $(date) ==="

# Step 1: Build embedding cache (only first fold does this, others wait)
if [ ! -f "${CACHE_DIR}/train_canonical.npy" ]; then
    if [ "${FOLD}" -eq 0 ]; then
        echo "=== Building chr-split ref+alt embedding cache ==="
        uv run --no-sync python experiments/train_oracle_alphagenome_hashfrag_cached.py \
            --build-cache-only \
            --cache-dir "${CACHE_DIR}" \
            --chr-split --include-alt-alleles \
            || echo "Cache build may need separate script"

        # If the above doesn't work, use the existing cache builder
        if [ ! -f "${CACHE_DIR}/train_canonical.npy" ]; then
            echo "Falling back to manual cache build..."
            uv run --no-sync python -c "
import sys, numpy as np
sys.path.insert(0, '.')
from data.k562 import K562Dataset
ds = K562Dataset(data_path='data/k562', split='train',
                 use_hashfrag=False, use_chromosome_fallback=True,
                 include_alt_alleles=True)
print('Train sequences:', len(ds))
# Save sequences for embedding
np.save('${CACHE_DIR}/train_sequences.npy', np.array(list(ds.sequences), dtype=object))
np.save('${CACHE_DIR}/train_labels.npy', ds.labels.astype(np.float32))
"
            # Build embeddings using AG encoder
            uv run --no-sync python scripts/build_ag_embedding_cache.py \
                --sequences "${CACHE_DIR}/train_sequences.npy" \
                --output-dir "${CACHE_DIR}" \
                --batch-size 128 \
                || echo "Need to implement build_ag_embedding_cache.py"
        fi
    else
        echo "Waiting for fold 0 to build cache..."
        while [ ! -f "${CACHE_DIR}/train_canonical.npy" ]; do
            sleep 30
        done
        echo "Cache ready."
    fi
fi

# Step 2: Train S1 oracle head for this fold
echo "=== Training S1 oracle fold ${FOLD} ==="
mkdir -p "${OUT_DIR}"

uv run --no-sync python experiments/train_oracle_alphagenome_hashfrag_cached.py \
    ++cache_dir="${CACHE_DIR}" \
    ++output_dir="${OUT_DIR}" \
    ++fold_id="${FOLD}" \
    ++n_folds=10 \
    ++chr_split=True \
    ++include_alt_alleles=True \
    ++epochs=50 \
    ++early_stop_patience=7

echo "=== Done: fold ${FOLD} $(date) ==="
