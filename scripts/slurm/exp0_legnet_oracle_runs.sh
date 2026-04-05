#!/bin/bash
#SBATCH --job-name=exp0_legnet_oracle
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# dependency removed — oracle already complete; submit with --dependency if needed

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# Run Exp0 with LegNet oracle for all students
# Depends on LegNet oracle labels being generated (job 1276561)

ORACLE_DIR=outputs/oracle_legnet_k562_ensemble
if [ ! -f "$ORACLE_DIR/summary.json" ]; then
    echo "LegNet oracle not ready at $ORACLE_DIR/summary.json"
    exit 1
fi

for STUDENT in alphagenome_k562_s1 dream_rnn dream_cnn legnet; do
    echo "=== Running $STUDENT with LegNet oracle ==="
    uv run --no-sync python experiments/exp1_1_scaling.py \
        --task k562 --student "$STUDENT" --reservoir random \
        --oracle legnet \
        --training-sizes 3197 6395 15987 31974 63949 159871 319742 \
        --seed 42
done
