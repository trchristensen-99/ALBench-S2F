#!/bin/bash
# Gating watcher: poll until task3 retries + task6 all finished, then
# print top retry cells per arch (for manual YAML update) and submit
# the task5/6/7 finalize watcher.

set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F

EXPECTED_LEGNET=18
EXPECTED_DATTN=20
EXPECTED_TASK6=36
POLL=300
MAX_TRIES=180

echo "=== Gating: wait for task3 retries + task6 ==="
for ((try=0; try<MAX_TRIES; try++)); do
    n_legnet=$(find results/preflight/task3_retry_legnet_noaug -name result.json 2>/dev/null | wc -l)
    n_dattn=$(find results/preflight/task3_retry_dream_attn_rcshift -name result.json 2>/dev/null | wc -l)
    n_task6=$(find results/preflight/task6_parameterization -name result.json 2>/dev/null | wc -l)
    echo "  poll $try: legnet_retry=$n_legnet/$EXPECTED_LEGNET dattn_retry=$n_dattn/$EXPECTED_DATTN task6=$n_task6/$EXPECTED_TASK6"
    if [ "$n_legnet" -ge "$EXPECTED_LEGNET" ] && [ "$n_dattn" -ge "$EXPECTED_DATTN" ] && [ "$n_task6" -ge "$EXPECTED_TASK6" ]; then
        echo "  threshold met"
        break
    fi
    sleep $POLL
done

echo
echo "=== Top retry cells per arch ==="
uv run --no-sync python scripts/preflight/_dump_task3_retry.py 2>&1 || true

echo
echo "=== Submit task5/6/7 finalize watcher (resumes chain) ==="
sbatch /grid/wsbs/home_norepl/christen/ALBench-S2F/scripts/preflight/_resubmit_task567_finalize.sh
echo "Done. Manually update pre_flight_decisions.yaml LR/BS for legnet+dream_attn if retry winners differ."
