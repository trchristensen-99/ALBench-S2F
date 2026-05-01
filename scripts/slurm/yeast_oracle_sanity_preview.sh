#!/bin/bash
# Sanity-check the ensemble oracle's behavior on the test split BEFORE
# running full pseudolabel generation. Uses currently-available 5 DRNN
# + 3 DCNN ckpts. Test set is small (71K) so this finishes in ~10-15 min.
#
#SBATCH --job-name=yeast_oracle_sanity
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=01:00:00
#SBATCH --mem=64G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export TORCHDYNAMO_DISABLE=1
source scripts/slurm/setup_hpc_deps.sh

uv run --no-sync python scripts/yeast_oracle_sanity_preview.py

echo "=== DONE — $(date) ==="
