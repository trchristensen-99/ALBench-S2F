#!/bin/bash
# Empirically characterize the HP-search rounds knee across all existing per-round
# curves (Step-1 bake-off + long-horizon rounds-scaling) to set the early-stopping
# criteria. Read-heavy (thousands of npz) -> compute node, not login.
#
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/analyze_rounds_knee.sh
#
#SBATCH --job-name=rounds_knee
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -uo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TQDM_DISABLE=1

echo "=== rounds_knee analysis $(date) ==="
uv run --no-sync python scripts/analysis/analyze_rounds_knee.py \
    --out_dir outputs/analysis_figures/rounds_knee
echo "=== done $(date) ==="
