#!/bin/bash
# Polls for Task 9 (D_min confirmation, locked HPs) to complete; runs
# the analyzer (writes d_min.confirmed); then runs task10_finalize.py
# in --dry-run mode so the validation report lands in the log without
# actually signing off — that final step waits for a human reviewer.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

EXPECTED=36   # 3 archs × 4 D × 3 seeds
POLL_SLEEP=300
MAX_HOURS=24
MAX_TRIES=$((MAX_HOURS * 3600 / POLL_SLEEP))

echo "=== Task 9 finalize: polling for $EXPECTED result.json files ==="
mkdir -p "$REPO/results/preflight/task9_d_min_confirm"
for ((try=0; try<MAX_TRIES; try++)); do
    n=$(find "$REPO/results/preflight/task9_d_min_confirm" -name 'result.json' 2>/dev/null | wc -l || echo 0)
    echo "  poll $try: $n result.json files"
    if [ "$n" -ge "$EXPECTED" ]; then
        echo "  threshold met; proceeding"
        break
    fi
    sleep "$POLL_SLEEP"
done

echo "=== Running analyze_task9_d_min_confirm ==="
uv run --no-sync python scripts/preflight/analyze_task9_d_min_confirm.py

echo "=== Running task10_finalize (dry-run) for validation report ==="
uv run --no-sync python scripts/preflight/task10_finalize.py --dry-run || true

echo
echo "=== Pre-flight pipeline complete (autonomous portion). ==="
echo "Final sign-off requires a human reviewer:"
echo "  uv run --no-sync python scripts/preflight/task10_finalize.py --reviewer YOUR_NAME"
echo "After sign-off, the YAML is treated as immutable for the main sweep."
