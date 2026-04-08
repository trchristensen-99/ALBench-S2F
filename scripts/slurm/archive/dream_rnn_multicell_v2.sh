#!/bin/bash
# Re-run DREAM-RNN on HepG2 and SK-N-SH with REAL (ground_truth) labels.
# Array 0-5: 3 seeds x 2 cell lines.
#
# Array mapping:
#   0-2: HepG2 seeds 0, 1, 2
#   3-5: SK-N-SH seeds 0, 1, 2
#
# Usage:
#   sbatch --array=0-5 scripts/slurm/dream_rnn_multicell_v2.sh
#
#SBATCH --job-name=drnn_multicell_v2
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

CELLS=("hepg2" "hepg2" "hepg2" "sknsh" "sknsh" "sknsh")
SEEDS=(0 1 2 0 1 2)

CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "=== DREAM-RNN Multicell v2 (ground_truth labels) ==="
echo "Cell: ${CELL}, Seed: ${SEED}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python experiments/exp1_1_scaling.py \
    --task k562 \
    --student dream_rnn \
    --oracle ground_truth \
    --cell-line "${CELL}" \
    --reservoir genomic \
    --n-replicates 1 \
    --no-hp-sweep \
    --seed "${SEED}" \
    --output-dir "outputs/dream_rnn_${CELL}_3seeds/seed_${SEED}" \
    --training-sizes 319742 \
    --epochs 80 \
    --ensemble-size 3 \
    --early-stop-patience 10

echo "Done: $(date)"
