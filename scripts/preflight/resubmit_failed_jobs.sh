#!/bin/bash
# Resubmit all jobs that failed during the disk-quota incident.
# All scripts were validated before the disk crash — just re-run them.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

echo "=== Resubmitting failed sweeps ==="

# Debias sweeps that crashed mid-training
bash scripts/preflight/launch_debias_v10_c91_refine.sh
bash scripts/preflight/launch_debias_v11_alt_modes.sh
bash scripts/preflight/launch_debias_v12_grid.sh
bash scripts/preflight/launch_debias_v13_extreme.sh
bash scripts/preflight/launch_debias_v14_axes.sh
bash scripts/preflight/launch_debias_v15_random.sh

# HP sweeps that crashed
bash scripts/preflight/launch_hp_edge_v3.sh
bash scripts/preflight/launch_hp_d300k_fill.sh
bash scripts/preflight/launch_v100_legnet_bonus.sh

# c91 and c86 10-fold
bash scripts/preflight/launch_c86_10fold.sh
bash scripts/preflight/launch_c91_10fold.sh

# Shift aug sweep + v16 (max_shift × c91)
bash scripts/preflight/launch_shift_aug_sweep.sh

echo
echo "=== Final queue ==="
/cm/shared/apps/slurm/current/bin/squeue -u christen --format="%.10i %.30j %.10T" 2>&1 | wc -l
echo " jobs in queue"
