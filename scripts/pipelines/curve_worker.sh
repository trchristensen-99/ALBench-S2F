#!/bin/bash
# A long-lived worker that pulls unclaimed curve points until the list is done.
#
# WHY NOT ANOTHER ARRAY. The high-priority tiers cap SUBMITTED jobs
# (QOSMaxSubmitJobPerUserLimit), so a 284-task array is rejected there outright. A
# handful of workers costs a handful of submissions and keeps those GPUs busy
# continuously, which is strictly better than an array that cannot be submitted.
#
# Safe alongside the slow_nice array because both take the same atomic claim: mkdir
# is atomic on POSIX and works on NFS, where flock does not. A worker releases its
# claim on exit; a claim with no result older than 3h is stolen, so a preempted
# worker (SIGKILL skips the trap) does not strand its point.
set -uo pipefail
REPO="${ALBENCH_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO" || exit 1
PY="${ALBENCH_PYTHON:-uv run --no-sync python}"
JOBS="${CURVE_JOBS:-outputs/curves/jobs.txt}"
# Stop early enough to finish the point in hand rather than being killed mid-write.
DEADLINE=$(( $(date +%s) + ${WORKER_SECONDS:-10800} ))

done_n=0; skipped=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "worker: deadline reached"; break; }
  out=$(printf '%s' "$line" | sed -n 's/.*--output-dir \([^ ]*\).*/\1/p')
  [ -z "$out" ] && continue
  if [ -n "$(find "$out" -name result.json -size +0c -print -quit 2>/dev/null)" ]; then
    skipped=$((skipped+1)); continue
  fi
  mkdir -p "$out"
  if ! mkdir "$out/.claim" 2>/dev/null; then
    if [ -n "$(find "$out/.claim" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then
      echo "worker: stealing stale claim $out"; rmdir "$out/.claim" 2>/dev/null
      mkdir "$out/.claim" 2>/dev/null || { skipped=$((skipped+1)); continue; }
    else
      skipped=$((skipped+1)); continue
    fi
  fi
  echo "worker: running $out"
  # shellcheck disable=SC2086
  eval "$PY $line"
  rc=$?
  rmdir "$out/.claim" 2>/dev/null
  [ $rc -eq 0 ] && done_n=$((done_n+1)) || echo "worker: FAILED rc=$rc $out"
done < "$JOBS"
echo "worker finished: completed=${done_n} skipped=${skipped}"
