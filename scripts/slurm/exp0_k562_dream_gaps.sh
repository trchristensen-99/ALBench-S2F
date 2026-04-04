#!/bin/bash
#SBATCH --job-name=exp0_k562_dream_gaps
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# Fill K562 DREAM-RNN and DREAM-CNN gaps at medium sizes (where count < 6)

for STUDENT in dream_rnn dream_cnn; do
    echo "=== Running $STUDENT ==="
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student "$STUDENT" --reservoir random \
        --training-sizes 3197 15987 31974 63949 --seed 42
done
