#!/bin/bash
# Finalize the deploy spec once the D=300k greedy selection has run on all 9 pools.
# Submitted by watchdog_step1_bakeoff_d300k.sh with an afterany dependency on the
# greedy_deploy_d300k array, so it runs after the per-pool greedy_deploy.json land.
#
# Writes:
#   outputs/hp_step1_bakeoff/deploy_spec_d300000.json        (D=300k standalone)
#   outputs/hp_step1_bakeoff/deploy_spec_d30000_300000.json  (cross-D final N*)
#
# Manual:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/finalize_deploy_spec.sh
#
#SBATCH --job-name=finalize_deploy_spec
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -uo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
export TQDM_DISABLE=1

echo "=== finalize_deploy_spec $(date) ==="
echo "--- D=300k standalone ---"
uv run --no-sync python scripts/analysis/aggregate_greedy_deploy.py --d 300000 || echo "  (d300000 aggregate failed)"
echo "--- cross-D (30k + 300k) FINAL N* ---"
uv run --no-sync python scripts/analysis/aggregate_greedy_deploy.py --d 30000,300000 || echo "  (cross-D aggregate failed)"
echo "=== finalize done $(date) ==="
