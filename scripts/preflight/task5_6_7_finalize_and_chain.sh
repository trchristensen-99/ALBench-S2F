#!/bin/bash
# Polls for Tasks 5, 6, 7 to complete; runs their analyzers (which lock
# augmentations / architecture_size / dropout in the YAML); then submits
# Task 9 (D_min confirmation with locked HPs).
#
# After Task 9 lands, ``task9_finalize_and_chain.sh`` runs analyze_task9
# + task10_finalize. The pipeline is then fully autonomous through final
# sign-off (modulo the human reviewer arg on task10).
#
# Submitted by task4_finalize_and_chain.sh once Tasks 5/6/7 have been
# launched.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# Task counts: 5=24 runs (4 augs × 3 archs × 2 seeds), 6=36 runs
# (3 archs × 3 sizes × 2 D × 2 seeds), 7=18 runs (3 archs × 3 dropouts × 2 seeds).
# Total expected: 78 result.json files across the three task dirs.
EXPECTED_5=24
EXPECTED_6=36
EXPECTED_7=18

POLL_SLEEP=600    # 10 min between checks (each run is multi-hour)
MAX_HOURS=72
MAX_TRIES=$((MAX_HOURS * 3600 / POLL_SLEEP))

echo "=== Tasks 5/6/7 finalize: polling for $((EXPECTED_5 + EXPECTED_6 + EXPECTED_7)) result.json files ==="
mkdir -p "$REPO/results/preflight/task5_augmentations"
mkdir -p "$REPO/results/preflight/task6_parameterization"
mkdir -p "$REPO/results/preflight/task7_dropout"
for ((try=0; try<MAX_TRIES; try++)); do
    n5=$(find "$REPO/results/preflight/task5_augmentations" -name 'result.json' 2>/dev/null | wc -l || echo 0)
    n6=$(find "$REPO/results/preflight/task6_parameterization" -name 'result.json' 2>/dev/null | wc -l || echo 0)
    n7=$(find "$REPO/results/preflight/task7_dropout" -name 'result.json' 2>/dev/null | wc -l || echo 0)
    echo "  poll $try: task5=$n5/$EXPECTED_5  task6=$n6/$EXPECTED_6  task7=$n7/$EXPECTED_7"
    if [ "$n5" -ge "$EXPECTED_5" ] && [ "$n6" -ge "$EXPECTED_6" ] && [ "$n7" -ge "$EXPECTED_7" ]; then
        echo "  thresholds met; proceeding"
        break
    fi
    sleep "$POLL_SLEEP"
done

echo "=== Running Task 5 / 6 / 7 analyzers ==="
uv run --no-sync python scripts/preflight/analyze_task5_augmentations.py
uv run --no-sync python scripts/preflight/analyze_task6_parameterization.py
uv run --no-sync python scripts/preflight/analyze_task7_dropout.py

echo "=== Running diagnostic plots ==="
uv run --no-sync python analysis/preflight.py

echo "=== Submitting Task 9 D_min confirmation (locked HPs) ==="
bash scripts/preflight/task9_d_min_confirm.sh

echo "=== Submitting Task 9 finalize watcher ==="
T9_FIN=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable <<'EOF'
#!/bin/bash
#SBATCH --job-name=pf_task9_finalize
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=8G
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
bash scripts/preflight/task9_finalize_and_chain.sh
EOF
)
echo "  task9_finalize submitted as $T9_FIN"
