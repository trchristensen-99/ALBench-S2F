#!/bin/bash
# Evaluate 4 new oracle neg-aug configs on random DNA, shuffled controls,
# and Agarwal intergenic sequences.
#
# Configs evaluated:
#   - var_tight_i5d2
#   - var_ar1a_w2x
#   - var_d20_tight
#   - var_10pct_real_inter
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/eval_neg_sweep_random_dna.sh
#
#SBATCH --job-name=neg_eval_rand
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=02:00:00
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
export ALPHAGENOME_WEIGHTS="/grid/wsbs/home_norepl/christen/alphagenome_weights/alphagenome-jax-all_folds-v1"

echo "=== Neg-sweep random DNA evaluation — $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run --no-sync python scripts/eval_neg_sweep_random_dna.py

echo "=== DONE — $(date) ==="
