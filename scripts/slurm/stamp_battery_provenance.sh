#!/bin/bash
# Stamp data/k562/test_sets_ag_s2_chrsplit/PROVENANCE.json with the canonical
# oracle id + battery version. CPU-only (reads npz headers, writes JSON) — no GPU.
#
# Launch (after the battery re-score + snv-mono jobs finish):
#   export PATH=/cm/shared/apps/slurm/current/bin:$PATH
#   sbatch --dependency=afterok:<snv_mono_jobid> scripts/slurm/stamp_battery_provenance.sh
#SBATCH --job-name=stamp_prov
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export TQDM_DISABLE=1

uv run --no-sync python scripts/stamp_battery_provenance.py
echo "=== PROVENANCE STAMPED — $(date) ==="
cat data/k562/test_sets_ag_s2_chrsplit/PROVENANCE.json
