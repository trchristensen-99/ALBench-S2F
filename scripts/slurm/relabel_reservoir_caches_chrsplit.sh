#!/bin/bash
# Re-label all 16 reservoir caches with the chr-split AG_S2 oracle ensemble.
#
# Background: the existing outputs/reservoir_cache/k562_{strategy}_d1000000_seed42.npz
# files were generated May 24-25 with _load_oracle("k562","ag_s2") which at that time
# loaded outputs/stage2_k562_oracle/ — a hashfrag-trained ensemble (confirmed by the
# "hashfrag" substring in the head name and by the deprecated train_stage2_k562_hashfrag.py
# training script). For the chr-split poster comparison we must regenerate them against
# outputs/oracle_chrsplit_natural/s2/ once all 10 folds are present.
#
# Each task ~25-40 min on H100 (1M predictions × 10-fold ensemble).
#
# Submit (with auto-trigger after retrain completes):
#   /cm/shared/apps/slurm/current/bin/sbatch \
#       --dependency=afterok:<retrain_array_id> \
#       --array=0-15 \
#       scripts/slurm/relabel_reservoir_caches_chrsplit.sh
#
#SBATCH --job-name=relabel_res
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

STRATEGIES=(
    random
    prm_1pct
    prm_5pct
    prm_10pct
    prm_20pct
    prm_attribution_1pct
    prm_uncertainty_1pct
    evoaug_heavy
    evoaug_structural
    motif_planted
    motif_planted_v2
    motif_shuffled
    motif_grammar
    phylogenetic_zoonomia
    dinuc_shuffle
    gc_matched
)

T=$SLURM_ARRAY_TASK_ID
S="${STRATEGIES[$T]}"
D=1000000
SEED=42
OUT="outputs/reservoir_cache/k562_${S}_d${D}_seed${SEED}.npz"
BAK="${OUT}.hashfrag_oracle_bak"

echo "=== relabel_res task=${T} strategy=${S} node=${SLURMD_NODENAME} $(date) ==="

# Safety: guard against falling back to legacy hashfrag oracle.
CHR_SPLIT_DIR="outputs/oracle_chrsplit_natural/s2"
N_FOLDS=$(find "${CHR_SPLIT_DIR}" -maxdepth 3 -type d -path "*/best_model/checkpoint" 2>/dev/null | wc -l)
if [ "${N_FOLDS}" -lt 10 ]; then
    echo "ERROR: chr-split oracle has only ${N_FOLDS}/10 folds — aborting to avoid hashfrag fallback"
    exit 2
fi
echo "Confirmed chr-split AG_S2: ${N_FOLDS}/10 folds present in ${CHR_SPLIT_DIR}"

# Backup the existing hashfrag-labeled cache before overwriting
if [ -f "${OUT}" ] && [ ! -f "${BAK}" ]; then
    mv "${OUT}" "${BAK}"
    echo "Backed up ${OUT} -> ${BAK}"
fi

uv run --no-sync python scripts/generate_reservoir_cache.py \
    --task k562 \
    --reservoir "${S}" \
    --D "${D}" \
    --seed "${SEED}" \
    --oracle ag_s2 \
    --out "${OUT}"

# Verify shape sanity before deleting the backup
python3 -c "
import numpy as np, sys
z = np.load('${OUT}', allow_pickle=True)
assert 'oracle_labels' in z.files, 'missing oracle_labels'
assert z['oracle_labels'].shape == (${D},), f'wrong shape: {z[\"oracle_labels\"].shape}'
print('OK: relabeled', z['oracle_labels'].shape, 'mean=', float(z['oracle_labels'].mean()))
" || { echo "Sanity check failed — keeping backup ${BAK}"; exit 3; }

# Remove the backup once we've confirmed the new cache is healthy
rm -f "${BAK}"

echo "=== Done: $(date) ==="
