#!/bin/bash
# Comprehensive Malinois evaluation with correct paper architecture.
#
# Runs paper-mode (correct arch + loss + optimizer) across:
#   1. Chr-split ref+alt (paper setting)
#   2. Chr-split ref-only (for comparison)
#   3. HashFrag split ref-only
#   4. Ablations: with/without Basset pretrained weights, with/without dup
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/malinois_comprehensive.sh
#
#SBATCH --job-name=mal_comp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Malinois comprehensive evaluation ==="
echo "Node: ${SLURMD_NODENAME} — $(date)"

# Ensure pretrained weights exist
uv run --no-sync python scripts/download_malinois_weights.py

run_malinois() {
    local NAME=$1
    local OUT_DIR=$2
    shift 2
    local EXTRA_ARGS=("$@")

    for SEED in 0 1 2; do
        RESULT="${OUT_DIR}/seed_${SEED}/seed_${SEED}/result.json"
        if [ -f "$RESULT" ]; then
            echo "  ${NAME} seed=${SEED}: already done"
            continue
        fi
        echo "  ${NAME} seed=${SEED} — $(date)"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++output_dir="${OUT_DIR}" \
            ++seed=${SEED} \
            ++paper_mode=True \
            "${EXTRA_ARGS[@]}"
    done
}

# ── 1. Chr-split ref+alt with pretrained (paper setting) ──────────────
echo ""
echo "--- 1. Chr-split ref+alt + pretrained conv weights ---"
run_malinois "chr_alt_pretrain" \
    "outputs/bar_final/k562/malinois_paper" \
    ++chr_split=True \
    ++pretrained_weights=data/pretrained/basset_pretrained.pkl

# ── 2. Chr-split ref+alt WITHOUT pretrained (ablation) ────────────────
echo ""
echo "--- 2. Chr-split ref+alt, no pretrained ---"
run_malinois "chr_alt_nopretrain" \
    "outputs/bar_final/k562/malinois_paper_nopretrain" \
    ++chr_split=True

# ── 3. Chr-split ref-only + pretrained ────────────────────────────────
echo ""
echo "--- 3. Chr-split ref-only + pretrained ---"
run_malinois "chr_ref_pretrain" \
    "outputs/chr_split/k562/malinois_paper" \
    ++chr_split=True \
    ++include_alt_alleles=False \
    ++pretrained_weights=data/pretrained/basset_pretrained.pkl

# ── 4. Chr-split ref-only, no pretrained ──────────────────────────────
echo ""
echo "--- 4. Chr-split ref-only, no pretrained ---"
run_malinois "chr_ref_nopretrain" \
    "outputs/chr_split/k562/malinois_paper_nopretrain" \
    ++chr_split=True \
    ++include_alt_alleles=False

# ── 5. HashFrag split + pretrained ────────────────────────────────────
echo ""
echo "--- 5. HashFrag + pretrained ---"
run_malinois "hf_pretrain" \
    "outputs/hashfrag/k562/malinois_paper" \
    ++chr_split=False \
    ++include_alt_alleles=False \
    ++pretrained_weights=data/pretrained/basset_pretrained.pkl

# ── 6. HashFrag, no pretrained ────────────────────────────────────────
echo ""
echo "--- 6. HashFrag, no pretrained ---"
run_malinois "hf_nopretrain" \
    "outputs/hashfrag/k562/malinois_paper_nopretrain" \
    ++chr_split=False \
    ++include_alt_alleles=False

# ── 7. Chr-split ref+alt + pretrained + shift augmentation ───────────
echo ""
echo "--- 7. Chr-split ref+alt + pretrained + shift aug ---"
run_malinois "chr_alt_shift" \
    "outputs/systematic_comparison/malinois_paper/shift" \
    ++chr_split=True \
    ++pretrained_weights=data/pretrained/basset_pretrained.pkl \
    ++shift_aug=True \
    ++max_shift=15

# ── 8. Chr-split ref+alt + pretrained, no duplication (ablation) ─────
echo ""
echo "--- 8. Chr-split ref+alt + pretrained, no dup ---"
# paper_mode sets dup_cutoff=0.5, override it
uv run --no-sync python -c "
import sys; sys.argv = ['']
from experiments.train_malinois_k562 import train_malinois, DEFAULT_CONFIG, PAPER_OVERRIDES
cfg = dict(DEFAULT_CONFIG)
cfg.update(PAPER_OVERRIDES)
cfg['chr_split'] = True
cfg['pretrained_weights'] = 'data/pretrained/basset_pretrained.pkl'
cfg['duplication_cutoff'] = None  # disable duplication
cfg['output_dir'] = 'outputs/systematic_comparison/malinois_paper/no_dup'
for seed in [0, 1, 2]:
    import os, json
    out = f\"outputs/systematic_comparison/malinois_paper/no_dup/seed_{seed}/seed_{seed}/result.json\"
    if os.path.exists(out):
        print(f'  no_dup seed={seed}: already done')
        continue
    cfg['seed'] = seed
    train_malinois(cfg)
"

# ── 9. Evaluate pretrained Malinois (re-run with fixed eval code) ─────
echo ""
echo "--- 9. Re-evaluate pretrained Malinois (fixed SNV/OOD) ---"
uv run --no-sync python scripts/eval_pretrained_malinois.py \
    --checkpoint data/pretrained/malinois_trained/torch_checkpoint.pt \
    --output-dir outputs/malinois_pretrained_eval_v2

echo ""
echo "=== All done — $(date) ==="
