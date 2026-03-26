#!/bin/bash
# DREAM-RNN single-model runs (ensemble_size=1) for fair comparison with
# non-ensembled models.  K562, HepG2, and SK-N-SH, 3 seeds each.
#
# Array mapping:
#   0-2: K562  seeds 0, 1, 2
#   3-5: HepG2 seeds 0, 1, 2
#   6-8: SKNSH seeds 0, 1, 2
#
# Usage:
#   sbatch --array=0-8 scripts/slurm/dream_rnn_single_model.sh
#
#SBATCH --job-name=drnn_single
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=12:00:00
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

CELLS=("k562" "k562" "k562" "hepg2" "hepg2" "hepg2" "sknsh" "sknsh" "sknsh")
SEEDS=(0 1 2 0 1 2 0 1 2)

CELL="${CELLS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "=== DREAM-RNN single-model (ensemble_size=1) ==="
echo "Cell: ${CELL}, Seed: ${SEED}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# Build command — K562 does not use --cell-line
CMD=(
    uv run --no-sync python experiments/exp1_1_scaling.py
    --task k562
    --student dream_rnn
    --oracle ground_truth
    --reservoir genomic
    --n-replicates 1
    --no-hp-sweep
    --seed "${SEED}"
    --output-dir "outputs/dream_rnn_${CELL}_single/seed_${SEED}"
    --training-sizes 319742
    --epochs 80
    --ensemble-size 1
    --early-stop-patience 10
)

if [[ "${CELL}" != "k562" ]]; then
    CMD+=(--cell-line "${CELL}")
fi

"${CMD[@]}"

echo "Done: $(date)"
