#!/bin/bash
# Task 4 auto-chain: poll for Task 4's 3 epoch-budget runs to complete,
# run analyze_task4 (locks epoch_budget per arch), then submit Tasks 5,
# 6, 7 in parallel.
#
# Submitted by task3_finalize_and_chain.sh (or run manually). Polls then
# fires Tasks 5/6/7 — these are parallelizable per the checklist.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

EXPECTED=3   # 1 run per arch at 240 epochs
TASK4_RESULTS="$REPO/results/preflight/task4_epoch_budget"
POLL_SLEEP=600     # 10 min between checks (each run is multi-hour)
MAX_HOURS=72
MAX_TRIES=$((MAX_HOURS * 3600 / POLL_SLEEP))

echo "=== Task 4 finalize: polling for $EXPECTED result.json files ==="
mkdir -p "$TASK4_RESULTS"
n=0
for ((try=0; try<MAX_TRIES; try++)); do
    n=$(find "$TASK4_RESULTS" -name 'result.json' 2>/dev/null | wc -l || echo 0)
    echo "  poll $try: $n result.json files"
    if [ "$n" -ge "$EXPECTED" ]; then
        echo "  threshold met; proceeding"
        break
    fi
    sleep "$POLL_SLEEP"
done
if [ "$n" -lt "$EXPECTED" ]; then
    echo "TIMEOUT: only $n files after ${MAX_HOURS}h. Aborting auto-chain."
    exit 1
fi

echo "=== Running analyze_task4_epoch_budget ==="
uv run --no-sync python scripts/preflight/analyze_task4_epoch_budget.py

echo "=== Submitting Tasks 5 / 6 / 7 in parallel ==="
bash scripts/preflight/task5_augmentations.sh
bash scripts/preflight/task6_parameterization.sh
bash scripts/preflight/task7_dropout.sh

echo "=== Submitting Tasks 5/6/7 finalize watcher (auto-fires Task 9 + analyzers) ==="
T567_FIN=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable <<'EOF'
#!/bin/bash
#SBATCH --job-name=pf_task567_finalize
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=8G
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
bash scripts/preflight/task5_6_7_finalize_and_chain.sh
EOF
)
echo "  task567_finalize submitted as $T567_FIN"

echo "=== Pre-flight chain set up through Task 9 + Task 10 dry-run validation. ==="
echo "Manual final step: uv run --no-sync python scripts/preflight/task10_finalize.py --reviewer NAME"
