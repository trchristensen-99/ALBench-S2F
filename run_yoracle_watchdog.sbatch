#!/bin/bash
#SBATCH --job-name=yo_wd
#SBATCH --partition=cpuq
#SBATCH --qos=slow_nice
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/yoracle_watchdog.out
set -uo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F; cd "$REPO"
SQ=/cm/shared/apps/slurm/current/bin/squeue; SB=/cm/shared/apps/slurm/current/bin/sbatch
STATUS=$REPO/outputs/yeast_oracle_v2/STATUS.txt
mkdir -p outputs/yeast_oracle_v2; resubs=0
note () { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }
note "yeast oracle watchdog start: variants A and B, 10 folds each"
while true; do
  alldone=1
  for v in A B; do
    # fold_[0-9] not fold_*: the latter also matches fold_0_B, so variant B folds
    # were counted toward variant A and the watchdog would have declared A complete
    # with folds still untrained.
    if [ "$v" = A ]; then pat="fold_[0-9]"; script=run_yoracle.sbatch; else pat="fold_[0-9]_B"; script=run_yoracleB.sbatch; fi
    d=$(ls outputs/yeast_oracle_v2/$pat/test_metrics.json 2>/dev/null | wc -l)
    # Count BOTH job names. Folds relaunched through scripts/slurm/launch_tiered.sh
    # are named yofold, not yoracle; matching only yoracle made the queue look empty
    # while nine folds were training, which would have resubmitted the whole array
    # on top of running jobs and had two processes writing one output directory.
    r=$($SQ -u christen -h -n yoracle,yoracleB,yofold 2>/dev/null | wc -l)
    note "variant $v: folds ${d}/10 trained, ${r} job(s) queued across all variants (resubmits: ${resubs})"
    [ "$d" -ge 10 ] && continue
    alldone=0
    if [ "$r" -eq 0 ]; then
      [ "$resubs" -ge 6 ] && { note "ERROR: resubmit budget exhausted (A/B at ${d}/10); see logs/yoracle/"; break 2; }
      note "queue empty with variant $v at ${d}/10 -> resubmitting $script (--resume skips finished folds)"
      $SB "$script" >> "$STATUS" 2>&1; resubs=$((resubs+1))
    fi
  done
  [ "$alldone" -eq 1 ] && { note "ALL 10 FOLDS DONE FOR BOTH VARIANTS"; break; }
  sleep 900
done
