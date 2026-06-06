#!/bin/bash
# Self-healing watchdog for the Step-1 deep bake-off (s1_* jobs).
#
# Re-runs the idempotent launcher (scripts/submit_step1_bakeoff.py) every 20 min.
# The launcher skips in-flight + already-complete (>=100 *_meta.json) cells and
# (re)submits any cell that is neither — i.e. anything that FAILED, TIMED OUT,
# was preempted, or exited cleanly for resume (LLM rate-limit pause). Each
# resubmitted cell resumes from its per-model *_meta.json checkpoints, so no
# completed compute is repeated.
#
# STEP1_FORCE_RAY=1 is exported so ray_asha/ray_bohb are NEVER skipped: this
# watchdog runs on a CPU node where `import ray` would fail the login-node probe,
# which previously left Ray cells silently un-resubmitted. The GPU jobs install
# ray via `uv run`, so forcing them on here is correct.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/watchdog_step1_bakeoff.sh
# Cancel:
#   scancel --name=s1_watchdog
#
#SBATCH --job-name=s1_watchdog
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --partition=cpuq
#SBATCH --qos=cpu_snice
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
PY="$REPO/.venv/bin/python"
cd "$REPO" || exit 1

export STEP1_FORCE_RAY=1   # CPU node has no ray; force-include ray cells anyway

INTERVAL=1200          # 20 min between launcher passes
MAX_PASSES=504         # 7 days / 20 min — hard stop matching walltime

echo "=== s1_watchdog start $(date) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} ==="
for ((pass=1; pass<=MAX_PASSES; pass++)); do
  echo "--- pass $pass $(date) ---"
  "$PY" "$REPO/scripts/submit_step1_bakeoff.py" 2>&1 | tail -8 || echo "  (launcher pass errored — will retry next interval)"
  remaining=$(squeue --me -h -o %j 2>/dev/null | grep -c '^s1_')
  # subtract self (job name s1_watchdog also matches ^s1_)
  remaining=$((remaining - 1))
  echo "  remaining s1_ bake-off jobs in queue: $remaining"
  if [ "$remaining" -le 0 ]; then
    # One confirming pass: if the launcher still queues nothing, all cells done.
    "$PY" "$REPO/scripts/submit_step1_bakeoff.py" 2>&1 | tail -3 || true
    remaining=$(($(squeue --me -h -o %j 2>/dev/null | grep -c '^s1_') - 1))
    if [ "$remaining" -le 0 ]; then
      echo "=== ALL STEP-1 BAKE-OFF CELLS COMPLETE — watchdog exiting $(date) ==="
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
echo "=== s1_watchdog reached MAX_PASSES; exiting $(date). Resubmit if bake-off incomplete. ==="
