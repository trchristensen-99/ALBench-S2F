#!/bin/bash
# Task 3 auto-chain: poll for the 50 Task 3 LR×BS D_max runs to complete,
# then run lock_task3_decisions + submit Task 3 verify + Task 4. Lets the
# preflight pipeline run autonomously through Task 4 without manual
# intervention.
#
# Wires into the existing dependency chain by being submitted by
# pf_task3_launcher (which, in turn, was scheduled afterok:aggregate).
#
# Sequence:
#   1. Wait until all expected Task 3 result.json files exist
#      (15 cells × 3 archs = 45 minimum; if DREAM-ATTN gets BS=128 it's 50).
#   2. Run analyze_hp_flatness (heatmap + flatness summary).
#   3. Run lock_task3_decisions (writes lr/batch_size to YAML).
#   4. Submit task3_verify_at_dmin.sh (18 runs, fast queue).
#   5. Submit task4_epoch_budget.sh (3 runs at 240 epochs each, slow_nice).

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

EXPECTED_MIN=45    # 15 cells × 3 archs (without DREAM-ATTN BS=128)
TASK3_RESULTS="$REPO/results/preflight/task3_lr_bs"
POLL_SLEEP=120     # 2 min between checks
MAX_HOURS=48
MAX_TRIES=$((MAX_HOURS * 3600 / POLL_SLEEP))

echo "=== Task 3 finalize: polling for $EXPECTED_MIN+ result.json files ==="
mkdir -p "$TASK3_RESULTS"
n=0
for ((try=0; try<MAX_TRIES; try++)); do
    # find can exit 1 if dir is missing or empty; tolerate that explicitly
    # because directory is created lazily by the first Task 3 result write.
    n=$(find "$TASK3_RESULTS" -name 'result.json' 2>/dev/null | wc -l || echo 0)
    echo "  poll $try: $n result.json files"
    if [ "$n" -ge "$EXPECTED_MIN" ]; then
        echo "  threshold met; proceeding"
        break
    fi
    sleep "$POLL_SLEEP"
done
if [ "$n" -lt "$EXPECTED_MIN" ]; then
    echo "TIMEOUT: only $n files after ${MAX_HOURS}h. Aborting auto-chain."
    exit 1
fi

echo "=== Running analyze_hp_flatness ==="
uv run --no-sync python scripts/preflight/analyze_hp_flatness.py || \
    echo "  WARN: analyze_hp_flatness failed; continuing."

echo "=== Running lock_task3_decisions ==="
uv run --no-sync python scripts/preflight/lock_task3_decisions.py

echo "=== Submitting Task 3 D_min verification ==="
bash scripts/preflight/task3_verify_at_dmin.sh

echo "=== Submitting Task 3 verify watcher (auto-runs analyze_task3_verify.py) ==="
T3V_FIN=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable <<'EOF'
#!/bin/bash
#SBATCH --job-name=pf_task3_verify_finalize
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=4G
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
EXPECTED=12   # min: 3 archs × 2 LR neighbors × 2 seeds = 12 (boundary archs)
DIR=results/preflight/task3_verify_dmin
mkdir -p "$DIR"
for try in $(seq 0 144); do
    n=$(find "$DIR" -name 'result.json' 2>/dev/null | wc -l || echo 0)
    echo "  poll $try: $n result.json files"
    if [ "$n" -ge "$EXPECTED" ]; then break; fi
    sleep 300
done
uv run --no-sync python scripts/preflight/analyze_task3_verify.py || \
    echo "WARN: analyze_task3_verify exit code $? — may indicate scale-coupling failure"
EOF
)
echo "  task3_verify_finalize submitted as $T3V_FIN"

echo "=== Submitting Task 4 epoch-budget calibration ==="
bash scripts/preflight/task4_epoch_budget.sh

echo "=== Submitting Task 4 finalize watcher (auto-fires Tasks 5/6/7 after Task 4 lands) ==="
T4_FIN=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable <<'EOF'
#!/bin/bash
#SBATCH --job-name=pf_task4_finalize
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpu_fill
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
#SBATCH --mem=8G
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
bash scripts/preflight/task4_finalize_and_chain.sh
EOF
)
echo "  task4_finalize submitted as $T4_FIN"

echo "=== Task 3 chain complete. Pipeline now autonomous through Tasks 5/6/7. ==="
echo "After Tasks 5+7 land, submit task9_d_min_confirm.sh (D_min with locked HPs)."
