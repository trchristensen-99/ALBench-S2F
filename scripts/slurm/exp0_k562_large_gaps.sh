#!/bin/bash
# Fill Exp0 K562 gaps at large training sizes (n=159871, n=319742).
# These are the slowest fractions, consistently incomplete across models.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/exp0_k562_large_gaps.sh
#
#SBATCH --job-name=exp0_large
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

echo "=== Exp0 K562 large-fraction gaps — $(date) ==="
echo "Node: ${SLURMD_NODENAME}"

# Default AG oracle, from-scratch models, missing seeds at large fractions
for STUDENT in dream_rnn dream_cnn legnet; do
    echo ""
    echo "--- ${STUDENT} default oracle (n=159871, n=319742) ---"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student "${STUDENT}" \
        --reservoir random \
        --training-sizes 159871 319742 \
        --seed 42 \
        || true
done

# Ground truth, from-scratch models
for STUDENT in dream_rnn dream_cnn legnet; do
    echo ""
    echo "--- ${STUDENT} ground truth (n=159871, n=319742) ---"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student "${STUDENT}" \
        --reservoir genomic \
        --oracle ground_truth \
        --training-sizes 159871 319742 \
        --seed 42 \
        || true
done

# DREAM-RNN oracle, from-scratch models
for STUDENT in dream_rnn dream_cnn legnet; do
    echo ""
    echo "--- ${STUDENT} oracle_dream_rnn (n=159871, n=319742) ---"
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 \
        --student "${STUDENT}" \
        --reservoir random \
        --oracle dream_rnn \
        --training-sizes 159871 319742 \
        --seed 42 \
        || true
done

echo ""
echo "=== Done — $(date) ==="
