#!/bin/bash
# Keep the curve chain alive overnight.
#
# The chain uses afterok dependencies, which means a preempted stage cancels everything
# downstream. slow_nice IS preemptible, so that will happen. This watchdog notices a
# stalled chain -- no curve jobs queued, work still outstanding -- and restarts from
# whichever stage is actually incomplete. Every stage is idempotent (finished pools and
# finished curve points are skipped), so restarting is always safe.
set -uo pipefail
REPO="${ALBENCH_REPO:-/grid/wsbs/home_norepl/christen/ALBench-S2F}"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
[ -f scripts/cluster/site.env ] && . scripts/cluster/site.env
SQ="${ALBENCH_SQUEUE:-$(command -v squeue)}"
PY="${ALBENCH_PYTHON:-uv run --no-sync python}"
STATUS="$REPO/outputs/curves/STATUS.txt"
note () { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

restarts=0
note "curve watchdog start"
while true; do
  sleep 900
  running=$($SQ -u "$USER" -h -n cv_gen,cv_verify_pools,cv_label,cv_verify_labels,cv_link,cv_train 2>/dev/null | wc -l)
  pools=$(ls outputs/curves/pools/*.npz 2>/dev/null | grep -vc labeled || echo 0)
  lab=$(ls outputs/curves/pools/*__labeled.npz 2>/dev/null | wc -l)
  pts=$(find outputs/curves/train -name result.json -size +0c 2>/dev/null | wc -l)
  want=$(wc -l < outputs/curves/jobs.txt 2>/dev/null || echo 0)
  note "pools=${pools} labelled=${lab} points=${pts}/${want} queued=${running} (restarts=${restarts})"

  [ "$want" -gt 0 ] && [ "$pts" -ge "$want" ] && { note "ALL CURVE POINTS DONE"; break; }
  if [ "$running" -eq 0 ]; then
    if [ "$restarts" -ge 8 ]; then
      note "ERROR: restart budget exhausted at ${pts}/${want}; see logs/curves/"
      break
    fi
    note "chain stalled with work outstanding -> resubmitting (finished work is skipped)"
    bash scripts/pipelines/curves_overnight.sh >> "$STATUS" 2>&1
    restarts=$((restarts + 1))
  fi
done
