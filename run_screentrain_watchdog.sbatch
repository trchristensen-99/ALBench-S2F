#!/bin/bash
#SBATCH --job-name=sct_wd
#SBATCH --partition=cpuq
#SBATCH --qos=slow_nice
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=48:00:00
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/screentrain/watchdog.out
# Resubmit the screen-training array if it empties before all 105 cells have results.
# slow_nice is preemptible, so tasks can vanish; run_screentrain.sbatch skips any cell
# that already has results.json, which makes blind resubmission safe.
set -uo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F; cd "$REPO"
SQ=/cm/shared/apps/slurm/current/bin/squeue; SB=/cm/shared/apps/slurm/current/bin/sbatch
STATUS=$REPO/outputs/screen/TRAIN_STATUS.txt
TOTAL=$(wc -l < outputs/screen/train_cells.txt)
resubs=0
note () { echo "[$(date "+%F %T")] $*" | tee -a "$STATUS"; }
note "screen-training watchdog start: ${TOTAL} cells"
while true; do
  d=$(find outputs/screen/train -name results.json -size +0c 2>/dev/null | wc -l)
  r=$($SQ -u christen -h -n sctrain 2>/dev/null | wc -l)
  note "cells ${d}/${TOTAL} trained, ${r} task(s) queued (resubmits: ${resubs})"
  [ "$d" -ge "$TOTAL" ] && { note "ALL ${TOTAL} SCREEN CELLS TRAINED"; break; }
  if [ "$r" -eq 0 ]; then
    [ "$resubs" -ge 8 ] && { note "ERROR: resubmit budget exhausted at ${d}/${TOTAL}"; break; }
    note "queue empty at ${d}/${TOTAL} -> resubmitting (finished cells are skipped)"
    $SB run_screentrain.sbatch >> "$STATUS" 2>&1; resubs=$((resubs+1))
  fi
  sleep 900
done
