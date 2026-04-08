#!/bin/bash
# Train DREAM-RNN K562 with ensemble_size=3 (3 seeds).
# The existing dream_rnn_k562_single_v2 has ensemble_size=1.
# This produces the 3-member ensemble results for fair comparison.
#
# Array: 0-2 (3 seeds)
#
#SBATCH --job-name=drnn_ens3
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
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
SEEDS=(42 123 456)
SEED="${SEEDS[$T]}"

echo "=== DREAM-RNN K562 ens3 seed=${SEED} node=${SLURMD_NODENAME} date=$(date) ==="

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 --student dream_rnn \
    --oracle ground_truth --reservoir genomic \
    --n-replicates 1 --seed "${SEED}" \
    --output-dir "outputs/dream_rnn_k562_3seeds/seed_${SEED}" \
    --training-sizes 319742 --epochs 80 \
    --ensemble-size 3 --early-stop-patience 10

echo "=== Done: $(date) ==="
