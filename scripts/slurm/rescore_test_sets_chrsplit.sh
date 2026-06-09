#!/bin/bash
# Re-score genomic / SNV / OOD / ctrl_neg test sets with the canonical full-pool
# random 10-fold AG_S2 oracle (full856k_clean), overriding AG_S2_ORACLE_DIR.
# Overwrites data/k562/test_sets_ag_s2_chrsplit/{genomic,snv,ood,ctrl_neg}_oracle.npz.
#
# Submit (auto-trigger after retrain):
#   /cm/shared/apps/slurm/current/bin/sbatch \
#       --dependency=afterok:<retrain_array_id> \
#       scripts/slurm/rescore_test_sets_chrsplit.sh
#
#SBATCH --job-name=rescore_tests
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
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

# Score the WHOLE battery with the canonical full-pool random 10-fold oracle
# (full856k_clean) so provenance matches the stamp. Set explicitly for every
# Python step below rather than relying on _load_oracle's default.
export AG_S2_ORACLE_DIR="${AG_S2_ORACLE_DIR:-$PWD/outputs/oracle_full856k_clean/s2}"

echo "=== rescore_tests node=${SLURMD_NODENAME} $(date) ==="
echo "Oracle: ${AG_S2_ORACLE_DIR}"

# Safety: ensure all 10 canonical-oracle folds are in place
N_FOLDS=$(find "${AG_S2_ORACLE_DIR}" -maxdepth 3 -type d -path "*/best_model/checkpoint" 2>/dev/null | wc -l)
if [ "${N_FOLDS}" -lt 10 ]; then
    echo "ERROR: canonical oracle only has ${N_FOLDS}/10 folds at ${AG_S2_ORACLE_DIR} — aborting"
    exit 2
fi
echo "Confirmed canonical AG_S2: ${N_FOLDS}/10 folds"

# Back up the hashfrag-labeled test set files
TEST_DIR="data/k562/test_sets_ag_s2_chrsplit"
RESCORE_LIST=(
    genomic_oracle.npz snv_oracle.npz ood_oracle.npz
    random_32k_oracle.npz dinuc_shuffle_oracle.npz
    sub_low_oracle.npz sub_med_oracle.npz sub_high_oracle.npz
    ins_low_oracle.npz ins_med_oracle.npz ins_high_oracle.npz
    del_low_oracle.npz del_med_oracle.npz del_high_oracle.npz
    translocation_oracle.npz inversion_oracle.npz
)
for fn in "${RESCORE_LIST[@]}"; do
    if [ -f "${TEST_DIR}/${fn}" ] && [ ! -f "${TEST_DIR}/${fn}.hashfrag_oracle_bak" ]; then
        cp "${TEST_DIR}/${fn}" "${TEST_DIR}/${fn}.hashfrag_oracle_bak"
        echo "Backed up ${fn}"
    fi
done

# 1. genomic / snv / ood — uses existing generate_ag_s2_test_labels.py
rm -f "${TEST_DIR}/genomic_oracle.npz" "${TEST_DIR}/snv_oracle.npz" "${TEST_DIR}/ood_oracle.npz"
uv run --no-sync python scripts/generate_ag_s2_test_labels.py

# 2. Mutagenesis battery (random_32k, dinuc, sub/ins/del × low/med/high,
#    translocation, inversion) — uses build_comprehensive_test_sets.py
rm -f "${TEST_DIR}/random_32k_oracle.npz" \
      "${TEST_DIR}/dinuc_shuffle_oracle.npz" \
      "${TEST_DIR}"/sub_*_oracle.npz \
      "${TEST_DIR}"/ins_*_oracle.npz \
      "${TEST_DIR}"/del_*_oracle.npz \
      "${TEST_DIR}/translocation_oracle.npz" \
      "${TEST_DIR}/inversion_oracle.npz"
uv run --no-sync python scripts/build_comprehensive_test_sets.py --task k562 --n 32000 --seed 42

# 3. ctrl_neg — uses score_ctrl_neg_ag_s2.py
uv run --no-sync python scripts/score_ctrl_neg_ag_s2.py

# Sanity check
python3 -c "
import numpy as np
from pathlib import Path
TD = Path('${TEST_DIR}')
checklist = [
    ('genomic_oracle.npz','oracle_mean'),
    ('snv_oracle.npz','ref_mean'),
    ('ood_oracle.npz','oracle_mean'),
    ('ctrl_neg_oracle.npz','oracle_mean'),
    ('random_32k_oracle.npz','oracle_mean'),
    ('dinuc_shuffle_oracle.npz','oracle_mean'),
    ('sub_high_oracle.npz','oracle_mean'),
    ('translocation_oracle.npz','oracle_mean'),
]
for fn, key in checklist:
    z = np.load(TD/fn, allow_pickle=True)
    a = z[key].astype(np.float32)
    print(f'{fn:30s} n={len(a):>6,}  mean={a.mean():+.3f}  std={a.std():.3f}')
"

# Cleanup backups on success
for fn in "${RESCORE_LIST[@]}"; do
    rm -f "${TEST_DIR}/${fn}.hashfrag_oracle_bak"
done

echo "=== Done: $(date) ==="
