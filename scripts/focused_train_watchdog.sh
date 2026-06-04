#!/bin/bash
# Watchdog: runs on HPC, periodically re-submits any failed/missing focused_train jobs.
# Survives VPN drops because it lives entirely on the HPC.
#
# Run as:  nohup bash scripts/focused_train_watchdog.sh > logs/focused_watchdog.log 2>&1 &
#          disown
#
# It loops every 20 minutes:
#   1. Re-runs submit_focused_train.py (which skips cells with model.npz or in-flight jobs)
#   2. Logs counts of submitted / skipped / queue state / completed
#   3. Exits cleanly when all 246 cells have model.npz

set -uo pipefail
REPO="/grid/wsbs/home_norepl/christen/ALBench-S2F"
cd "$REPO"

# Total expected: 6 reservoirs × 6-7 D × 3 configs × 2 seeds
# (genomic 6 D, others 7 D, all × 3 configs × 2 seeds)
TOTAL=$((1*6*3*2 + 5*7*3*2))   # genomic + non-genomic = 36 + 210 = 246

echo "[watchdog] start $(date)  TOTAL_EXPECTED=$TOTAL"

while true; do
    NOW=$(date)
    DONE=$(find outputs/focused_train -name model.npz 2>/dev/null | wc -l)
    QUEUED=$(/cm/shared/apps/slurm/current/bin/squeue --me -h --format="%T %j" 2>/dev/null | grep -cE "ft_|^PENDING ft_|^RUNNING ft_" || echo 0)
    echo "[watchdog] $NOW  done=$DONE/$TOTAL  in_queue=$QUEUED"

    if [ "$DONE" -ge "$TOTAL" ]; then
        echo "[watchdog] ALL DONE. exiting."
        break
    fi

    # Re-run the launcher; it skips done cells and in-flight jobs automatically
    source .venv/bin/activate 2>/dev/null
    python3 scripts/submit_focused_train.py 2>&1 | grep -E "submitted|Skipped|ERR" | tail -5

    sleep 1200  # 20 minutes
done

echo "[watchdog] end $(date)"
