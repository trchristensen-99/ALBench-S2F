#!/bin/bash
# Generate Stage 2 AlphaGenome oracle ensemble pseudo-labels for K562 hashFrag.
#
# Stage 2 uses fine-tuned encoders (per fold), so full-encoder inference is
# required for every split (train+pool, val, all test sets).
#
# Runtime estimate: ~9-12 hours (10 folds × ~320K train+pool full-encoder
# inferences + test sets).
#
# Prerequisites:
#   1. 10 Stage 2 oracle checkpoints: outputs/stage2_k562_oracle/fold_{0..9}/best_model/
#   2. Test set TSVs:                 data/k562/test_sets/
#
#SBATCH --job-name=oracle_pseudolabels_stage2_k562_ag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

echo "Generating Stage 2 K562 AG oracle pseudolabels on $(date)"
echo "Node: ${SLURMD_NODENAME}"

uv run --no-sync python experiments/generate_oracle_pseudolabels_stage2_k562_ag.py \
    ++wandb_mode=disabled
