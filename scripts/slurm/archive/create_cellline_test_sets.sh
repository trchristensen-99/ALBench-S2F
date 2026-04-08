#!/bin/bash
# Create test set TSV files for HepG2 and SK-N-SH.
# CPU-only — just reads raw data + creates TSV files.
# Must run BEFORE dream_rnn_multicell_v2.sh (needs test sets for evaluation).
#
# Usage:
#   sbatch scripts/slurm/create_cellline_test_sets.sh
#
#SBATCH --job-name=create_test_sets
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

echo "=== Creating Cell-Line Test Sets ==="
echo "Node: $SLURMD_NODENAME  Date: $(date)"

# First ensure K562 test sets exist (needed as source)
if [[ ! -f "data/k562/test_sets/test_in_distribution_hashfrag.tsv" ]]; then
    echo "Creating K562 test sets first..."
    uv run --no-sync python scripts/create_k562_test_sets.py
fi

# Create HepG2 and SK-N-SH test sets
for CELL in hepg2 sknsh; do
    echo "--- Creating ${CELL} test sets ---"
    uv run --no-sync python scripts/create_cellline_test_sets.py --cell-line "${CELL}"
done

echo "Done: $(date)"
