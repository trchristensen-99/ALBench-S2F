#!/bin/bash
# Keep the claim of every RUNNING curve point looking fresh.
#
# The steal rule frees a claim older than a threshold so a preempted worker cannot
# strand its point. That threshold was set to 180 minutes before we knew how long the
# largest points take -- measured at up to 7h28 for a 284,981-sequence point. A live
# job therefore looks abandoned after three hours, and the next task to reach that
# entry would steal it and start a SECOND writer on the same output directory.
#
# The array already submitted carries the old threshold and cannot be edited in place,
# so instead of cancelling hours of in-progress training this touches the claim of
# every point that a running job actually holds. A claim being touched is never stale,
# so the rule stops firing on live work while still freeing genuinely dead claims.
set -uo pipefail
REPO="${ALBENCH_REPO:-/grid/wsbs/home_norepl/christen/ALBench-S2F}"
cd "$REPO" || exit 1
SQ="${ALBENCH_SQUEUE:-/cm/shared/apps/slurm/current/bin/squeue}"
JOBS=outputs/curves/jobs.txt
LOG=outputs/curves/KEEPALIVE.txt

while true; do
  n=0
  # array tasks: index maps to a line of the plan
  for j in $($SQ -u "$USER" -h -t RUNNING -n cv_train -o "%i" 2>/dev/null); do
    idx=${j##*_}
    case "$idx" in ''|*[!0-9]*) continue;; esac
    out=$(sed -n "${idx}p" "$JOBS" 2>/dev/null | sed -n 's/.*--output-dir \([^ ]*\).*/\1/p')
    [ -n "$out" ] && [ -d "$out/.claim" ] && { touch "$out/.claim"; n=$((n+1)); }
  done
  # workers: the point they are on is the last one their log reports
  for f in logs/curves/wdef-*.out logs/curves/wfast-*.out; do
    [ -f "$f" ] || continue
    j=$(basename "$f" .out); j=${j#wdef-}; j=${j#wfast-}
    [ -z "$($SQ -h -j "$j" -o "%T" 2>/dev/null)" ] && continue
    p=$(grep -hoE "outputs/curves/train/[a-z0-9_]+__base[0-9]+__add[0-9]+__s[0-9]+" "$f" 2>/dev/null | tail -1)
    [ -n "$p" ] && [ -d "$p/.claim" ] && { touch "$p/.claim"; n=$((n+1)); }
  done
  echo "[$(date '+%F %T')] refreshed ${n} live claims" >> "$LOG"
  sleep 300
done
