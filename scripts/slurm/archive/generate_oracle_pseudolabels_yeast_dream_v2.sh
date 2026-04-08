#!/bin/bash
# Generate DREAM-RNN yeast oracle pseudolabels v2 (from optimized ensemble).
#
# Uses the v2 oracle ensemble (oracle_dream_rnn_yeast_kfold_v2/) trained with
# optimized HPs (bs=512, lr=0.005, dropout_lstm=0.3).
#
# Must run AFTER train_oracle_dream_rnn_v2.sh completes all 10 folds.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/generate_oracle_pseudolabels_yeast_dream_v2.sh
#
#SBATCH --job-name=oracle_labels_v2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

echo "=== Generating yeast DREAM oracle pseudolabels v2 ==="
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/generate_oracle_pseudolabels_yeast_dream.py \
  --config-name generate_oracle_pseudolabels_yeast_dream \
  ++oracle_dir=outputs/oracle_dream_rnn_yeast_kfold_v2 \
  ++output_dir=outputs/oracle_pseudolabels/yeast_dream_oracle_v2 \
  ++dropout_lstm=0.3 \
  ++dropout_cnn=0.2

echo "=== DONE — $(date) ==="
