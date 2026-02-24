#!/bin/bash
# Evaluate AlphaGenome Boda heads (sum, mean, max, center) on chr 7, 13 test set.
# Writes outputs/ag_chrom_test_results.json for comparison with Malinois.
#SBATCH --job-name=ag_chrom_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

set -e
source /etc/profile.d/modules.sh
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD:$PYTHONPATH"

uv run python scripts/analysis/eval_ag_chrom_test.py \
    --data_path data/k562 \
    --output outputs/ag_chrom_test_results.json
