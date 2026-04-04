#!/bin/bash
# Re-evaluate Malinois paper-mode checkpoints with fixed multitask eval,
# then run remaining comprehensive conditions.
#
# Submit after mal_paper finishes:
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterany:JOB_ID scripts/slurm/malinois_reeval_and_comprehensive.sh
#
#SBATCH --job-name=mal_reeval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Malinois re-eval and comprehensive — $(date) ==="
echo "Node: ${SLURMD_NODENAME}"

# Ensure pretrained weights exist
uv run --no-sync python scripts/download_malinois_weights.py

# Step 1: Delete any broken result.json files (empty test_metrics from old code)
echo ""
echo "--- Cleaning broken result files ---"
for DIR in outputs/bar_final/k562/malinois_paper outputs/bar_final/k562/malinois_paper_nopretrain; do
    for f in ${DIR}/seed_*/result.json; do
        [ -f "$f" ] || continue
        # Check if test_metrics are empty
        EMPTY=$(python3 -c "
import json
d = json.load(open('$f'))
tm = d.get('test_metrics', {})
empty = all(not v for v in tm.values()) if tm else True
print('yes' if empty else 'no')
" 2>/dev/null)
        if [ "$EMPTY" = "yes" ]; then
            echo "  Deleting broken: $f"
            rm -f "$f"
        fi
    done
done

# Step 2: Re-evaluate pretrained Malinois (with fixed code)
echo ""
echo "--- Re-evaluating pretrained Malinois ---"
uv run --no-sync python scripts/eval_pretrained_malinois.py \
    --checkpoint data/pretrained/malinois_trained/torch_checkpoint.pt \
    --output-dir outputs/malinois_pretrained_eval_v2

# Step 3: Run comprehensive Malinois conditions
run_mal() {
    local NAME=$1; local OUTDIR=$2; shift 2
    for SEED in 0 1 2; do
        RESULT="${OUTDIR}/seed_${SEED}/result.json"
        [ -f "$RESULT" ] && echo "  ${NAME}/s${SEED}: done" && continue
        echo "  ${NAME}/s${SEED} — $(date)"
        uv run --no-sync python experiments/train_malinois_k562.py \
            ++output_dir="${OUTDIR}" ++seed=${SEED} ++paper_mode=True "$@"
    done
}

echo ""
echo "--- 1. Chr-split ref+alt + pretrained ---"
run_mal "chr_alt_pt" "outputs/bar_final/k562/malinois_paper" \
    ++chr_split=True ++pretrained_weights=data/pretrained/basset_pretrained.pkl

echo ""
echo "--- 2. Chr-split ref+alt, no pretrained ---"
run_mal "chr_alt_nopt" "outputs/bar_final/k562/malinois_paper_nopretrain" \
    ++chr_split=True

echo ""
echo "--- 3. Chr-split ref-only + pretrained ---"
run_mal "chr_ref_pt" "outputs/chr_split/k562/malinois_paper" \
    ++chr_split=True ++include_alt_alleles=False \
    ++pretrained_weights=data/pretrained/basset_pretrained.pkl

echo ""
echo "--- 4. Chr-split ref-only, no pretrained ---"
run_mal "chr_ref_nopt" "outputs/chr_split/k562/malinois_paper_nopretrain" \
    ++chr_split=True ++include_alt_alleles=False

echo ""
echo "--- 5. HashFrag + pretrained ---"
run_mal "hf_pt" "outputs/hashfrag/k562/malinois_paper" \
    ++chr_split=False ++include_alt_alleles=False \
    ++pretrained_weights=data/pretrained/basset_pretrained.pkl

echo ""
echo "--- 6. HashFrag, no pretrained ---"
run_mal "hf_nopt" "outputs/hashfrag/k562/malinois_paper_nopretrain" \
    ++chr_split=False ++include_alt_alleles=False

echo ""
echo "--- 7. Chr-split + shift aug ---"
run_mal "shift" "outputs/systematic_comparison/malinois_paper/shift" \
    ++chr_split=True ++pretrained_weights=data/pretrained/basset_pretrained.pkl \
    ++shift_aug=True ++max_shift=15

echo ""
echo "--- 8. Chr-split + no duplication ---"
run_mal "nodup" "outputs/systematic_comparison/malinois_paper/no_dup" \
    ++chr_split=True ++pretrained_weights=data/pretrained/basset_pretrained.pkl \
    ++duplication_cutoff=None

echo ""
echo "=== All done — $(date) ==="
