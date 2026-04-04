#!/bin/bash
#SBATCH --job-name=exp0_yeast_large
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

# Fill yeast Exp0 gaps at large fractions (n=1213065, n=3032661, n=6065324)
# These are very slow — needs slow_nice 48h QoS

for STUDENT in dream_rnn dream_cnn legnet; do
    echo "=== Running $STUDENT ==="
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task yeast --student "$STUDENT" --reservoir random \
        --training-sizes 1213065 3032661 6065324 --seed 42
done
