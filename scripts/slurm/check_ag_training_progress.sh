#!/bin/bash
# Run on HPC (e.g. after: cd /grid/wsbs/home_norepl/christen/ALBench-S2F) to monitor
# AlphaGenome training: job status, cache build, and correlation values.
# Usage: bash scripts/slurm/check_ag_training_progress.sh [job_id]
# If job_id is given, only that job's logs are parsed; otherwise all recent ag_* logs.

set -e
cd /grid/wsbs/home_norepl/christen/ALBench-S2F 2>/dev/null || cd "$(dirname "$0")/../.."
LOG_DIR="${LOG_DIR:-logs}"

echo "=== Job queue (christen) ==="
squeue -u christen 2>/dev/null || true

echo ""
echo "=== Recent AlphaGenome job logs (cache + val metrics) ==="
if [ -n "$1" ]; then
  PATTERN="*${1}*.out"
else
  PATTERN="ag_*.out"
fi

for f in $(ls -t "$LOG_DIR"/$PATTERN 2>/dev/null | head -20); do
  echo "--- $f ---"
  grep -E "\[EmbeddingCache\]|val/pearson|best_val_pearson|Early stopping|Compact window|Error|Traceback" "$f" 2>/dev/null || true
  echo ""
done

echo "=== Cache directories under outputs ==="
find outputs -maxdepth 3 -type d -name "embedding_cache*" 2>/dev/null || true
ls -la outputs/*/embedding_cache*/ 2>/dev/null || true
