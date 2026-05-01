#!/bin/bash
# Generate ensemble oracle pseudolabels for yeast train/val/test splits.
# Loads 5 DRNN + 5 DCNN ckpts, predicts on each split, averages.
# Output: outputs/yeast_ensemble_oracle/{train,val,test}_pseudolabels.npz
#
# Each DRNN inference on 6M = ~10-15 min on H100.
# Each DCNN inference on 6M = ~5-8 min.
# Total: 5*15 + 5*8 = ~115 min for train, smaller for val/test.
#
#SBATCH --job-name=yeast_ens_oracle
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

uv run --no-sync python scripts/generate_yeast_ensemble_oracle.py

echo "=== DONE — $(date) ==="
