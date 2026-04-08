#!/bin/bash
# Build OOD designed test sets for HepG2 and SK-N-SH.
# No GPU needed — CPU-only data processing.
#
# Usage:
#   sbatch scripts/slurm/build_multicell_ood_data.sh
#
#SBATCH --job-name=build_ood_data
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

set +u
source /etc/profile.d/modules.sh
set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh

MPRA_DIR="data/zenodo_10698014/MPRA_Datasets"

if [[ ! -d "${MPRA_DIR}" ]]; then
    echo "ERROR: MPRA directory not found: ${MPRA_DIR}"
    echo "Download the Zenodo dataset first. Expected path:"
    echo "  ${PWD}/${MPRA_DIR}"
    echo "See data/README.md or https://zenodo.org/record/10698014 for instructions."
    exit 1
fi

echo "=== Building Multicell OOD Designed Test Sets ==="
echo "MPRA dir: ${MPRA_DIR}"
echo "Node: $SLURMD_NODENAME  Date: $(date)"

uv run --no-sync python scripts/build_multicell_ood_designed.py \
    --mpra-dir "${MPRA_DIR}"

echo "Done: $(date)"
