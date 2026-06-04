#!/bin/bash
# Score the Gosai ctrl_neg subset with the chr-split AG_S2 oracle ensemble.
# Produces data/k562/test_sets_ag_s2_chrsplit/ctrl_neg_oracle.npz.
#
# MUST be submitted only after all 10 folds of oracle_chrsplit_natural/s2/
# have completed — _load_oracle() falls back to the legacy hashfrag-trained
# ensemble if fewer than 10 folds are present.
#
#SBATCH --job-name=score_ctrl_neg
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

set -euo pipefail

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

echo "=== score_ctrl_neg node=${SLURMD_NODENAME} $(date) ==="

uv run --no-sync python scripts/score_ctrl_neg_ag_s2.py

echo "=== Done: $(date) ==="
