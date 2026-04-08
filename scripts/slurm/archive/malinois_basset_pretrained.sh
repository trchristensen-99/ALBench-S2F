#!/bin/bash
# Train Malinois with Basset pretrained conv weights. 3 random seeds.
# Uses the best hyperparams from the baseline run.
# Pretrained weights are downloaded from Zenodo (Kelley et al. 2016).
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/malinois_basset_pretrained.sh
#
#SBATCH --job-name=malinois_basset
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

# Download pretrained weights if needed
WEIGHTS_DIR="data/pretrained"
WEIGHTS_PATH="${WEIGHTS_DIR}/basset_pretrained.pth"
if [ ! -f "${WEIGHTS_PATH}" ]; then
    echo "Downloading Basset pretrained weights..."
    mkdir -p "${WEIGHTS_DIR}"
    curl -sL "https://zenodo.org/record/1466068/files/pretrained_model_reloaded_th.pth?download=1" \
        -o "${WEIGHTS_PATH}"
    echo "Downloaded: $(ls -lh ${WEIGHTS_PATH})"
fi

echo "Malinois (Basset pretrained) K562: seed_idx=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}  Date: $(date)"

uv run --no-sync python experiments/train_malinois_k562.py \
    ++output_dir=outputs/malinois_k562_basset_pretrained \
    ++pretrained_weights="${WEIGHTS_PATH}"

echo "seed_idx=${SLURM_ARRAY_TASK_ID} DONE — $(date)"
