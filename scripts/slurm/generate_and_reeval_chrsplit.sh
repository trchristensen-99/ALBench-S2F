#!/bin/bash
# Two-step pipeline:
# Step 1: Generate AG S2 oracle labels for chr-split test sequences (H100, ~15 min)
# Step 2: Re-evaluate ALL existing LegNet models on chr-split AG S2 test sets (V100, ~30 min)
#
# Submit as dependency chain:
#   STEP1=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable scripts/slurm/generate_and_reeval_chrsplit.sh)
#   /cm/shared/apps/slurm/current/bin/sbatch --dependency=afterok:$STEP1 --array=0 scripts/slurm/generate_and_reeval_chrsplit.sh
#
# Or submit both steps as a single job (step 1 then step 2):
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/generate_and_reeval_chrsplit.sh
#
#SBATCH --job-name=chrsplit_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

echo "=== Step 1: Generate AG S2 chr-split test labels — $(date) ==="

TEST_DIR="data/k562/test_sets_ag_s2_chrsplit"
if [ -f "${TEST_DIR}/genomic_oracle.npz" ] && [ -f "${TEST_DIR}/snv_oracle.npz" ] && [ -f "${TEST_DIR}/ood_oracle.npz" ]; then
    echo "Test labels already exist, skipping generation."
else
    uv run --no-sync python scripts/generate_ag_s2_test_labels.py
fi

echo ""
echo "=== Step 2: Re-evaluate all LegNet+AG_S2 models — $(date) ==="

# Re-evaluate all results in the main output directory
uv run --no-sync python scripts/reeval_chrsplit_ag_s2.py \
    --results-dir outputs/exp1_1/k562/legnet_ag_s2 \
    --test-dir "${TEST_DIR}" \
    --force

echo ""
echo "=== Done — $(date) ==="
