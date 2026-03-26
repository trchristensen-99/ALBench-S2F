#!/bin/bash
# Re-evaluate SNV metrics for HepG2/SK-N-SH models using cell-specific SNV labels.
#
# The original training evaluations used K562 SNV labels as fallback.
# Now that data/{cell}/test_sets/test_snv_pairs_hashfrag.tsv exists with
# cell-specific columns, we re-evaluate and patch the existing result JSONs.
#
# Uses eval_ood_multicell.py infrastructure extended to patch SNV metrics.
#
# Usage:
#   sbatch scripts/slurm/reeval_multicell_snv.sh
#
#SBATCH --job-name=reeval_snv
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=4:00:00
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

echo "=== Re-evaluating SNV metrics for HepG2/SK-N-SH ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python scripts/reeval_multicell_snv.py

echo "Done: $(date)"
