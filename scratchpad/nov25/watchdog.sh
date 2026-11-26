#!/bin/bash
# Overnight watchdog: keep the queue full, catch failures, and leave a readable morning log.
#
# It runs as a batch job rather than on the login node, and it does three things every
# cycle. None of them submit new science -- a watchdog that invents work is a watchdog
# nobody can audit in the morning.
#
#   1. RECORD. One line per cycle with my running/pending counts and the cluster's free
#      GPUs by type, so the morning question "were we idle at 3am and why" has an answer
#      instead of a guess.
#   2. OPEN THE THROTTLE. If I am running fewer tasks than the target and the queue has
#      pending array tasks, raise their ArrayTaskThrottle. Arrays were submitted with
#      conservative throttles to be a good neighbour while the cluster was busy; overnight
#      the H100s free up and a %4 throttle just leaves them idle.
#   3. REPORT FAILURES. Any task that enters a terminal non-COMPLETED state gets its name,
#      exit code and first error line appended to a failures file. It does NOT resubmit:
#      every failure tonight has been a real bug, and a watchdog that quietly retries them
#      would have hidden four of them.
#
# The throttle ceiling stays under the slow_nice caps (20 h100, 20 v100, 40 jobs) so this
# cannot wedge the account against its own QOS limit.

set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
KOO=/grid/koo/home/christen/nov25_extra
LOG="$KOO/watchdog.log"
FAIL="$KOO/watchdog_failures.log"
TARGET=${TARGET:-24}      # tasks I want running; slow_nice allows 40 jobs
CEILING=${CEILING:-12}    # highest per-array throttle to set
SLEEP=${SLEEP:-600}
END=$(( $(date +%s) + ${HOURS:-14} * 3600 ))

mkdir -p "$KOO"
: > "$FAIL"
seen_fail=""
# Only failures from THIS watchdog's lifetime. The first version looked back fourteen
# hours and dutifully re-reported the afternoon's already-fixed bugs, so the morning's
# failure file opened with eight entries that were all stale -- which is worse than an
# empty one, because it buries anything genuinely new.
SINCE=$(date '+%Y-%m-%dT%H:%M')
echo "# watchdog started $(date '+%F %T'), target=$TARGET ceiling=$CEILING until $(date -d @$END '+%F %T' 2>/dev/null || echo '+14h')" >> "$LOG"

while [ "$(date +%s)" -lt "$END" ]; do
  run=$(squeue -u christen -h -t RUNNING 2>/dev/null | wc -l)
  pend=$(squeue -u christen -h -t PENDING 2>/dev/null | wc -l)
  # Free GPUs by type, from sinfo's own Gres/GresUsed per node.
  #
  # The previous version subtracted what `squeue -o '%b'` reported from the configured
  # total, and that undercounts: %b does not report every running job's GRES, so this log
  # claimed 4 free H100 for hours while sinfo's GresUsed showed 76 of 76 allocated. Jobs
  # were queued against capacity that did not exist. GresUsed is what the scheduler
  # actually holds, so it is what gets logged.
  read -r freeh freev < <(sinfo -p gpuq -h -N -O 'Gres:24,GresUsed:28' 2>/dev/null | awk '
    { tot=0; used=0; type="";
      if (match($0, /gpu:(h100|v100):[0-9]+/)) { s=substr($0,RSTART,RLENGTH); split(s,a,":"); type=a[2]; tot=a[3]; }
      if (match($0, /gpu:(h100|v100):[0-9]+\(IDX/)) { s=substr($0,RSTART,RLENGTH); split(s,b,":"); sub(/\(IDX/,"",b[3]); used=b[3]; }
      if (type != "") { T[type]+=tot; U[type]+=used }
    } END { printf "%d %d\n", T["h100"]-U["h100"], T["v100"]-U["v100"] }')
  freeh=${freeh:-0}; freev=${freev:-0}
  printf '%s running=%-3s pending=%-4s free_h100=%-4s free_v100=%-4s %s\n' \
    "$(date '+%F %T')" "$run" "$pend" "$freeh" "$freev" \
    "$(squeue -u christen -h -t RUNNING -o '%j' 2>/dev/null | sort | uniq -c | tr '\n' ' ')" >> "$LOG"

  # 2. Open throttles when we are under target and something is waiting.
  if [ "$run" -lt "$TARGET" ] && [ "$pend" -gt 0 ]; then
    for jid in $(squeue -u christen -h -t PENDING -o '%i' 2>/dev/null | grep '_\[' | cut -d_ -f1 | sort -u); do
      cur=$(squeue -j "$jid" -h -o '%i' 2>/dev/null | grep -oE '%[0-9]+' | tr -d '%' | head -1)
      [ -z "${cur:-}" ] && continue
      [ "$cur" -ge "$CEILING" ] && continue
      new=$(( cur * 2 )); [ "$new" -gt "$CEILING" ] && new=$CEILING
      if scontrol update JobId="$jid" ArrayTaskThrottle="$new" 2>/dev/null; then
        echo "$(date '+%F %T')   throttle $jid: $cur -> $new" >> "$LOG"
      fi
    done
  fi

  # 3. Report terminal failures once each, with the first error line from the log.
  while read -r id name st code; do
    case "$st" in
      COMPLETED|RUNNING|PENDING|COMPLETING|REQUEUED|RESIZING|SUSPENDED) continue ;;
    esac
    case "$seen_fail" in *"|$id|"*) continue ;; esac
    seen_fail="$seen_fail|$id|"
    f=$(ls -t "$KOO"/logs/*"${id%%_*}"*.log 2>/dev/null | head -1)
    err=$(grep -m1 -E 'Error|error:|AssertionError|refus|No such file|duplicate' "$f" 2>/dev/null | cut -c1-160)
    echo "$(date '+%F %T') $id $name $st $code :: ${err:-no error line found}" >> "$FAIL"
  done < <(sacct -u christen -S "$SINCE" \
                 -X -n -P -o JobID,JobName,State,ExitCode 2>/dev/null | tr '|' ' ')

  sleep "$SLEEP"
done
echo "# watchdog finished $(date '+%F %T')" >> "$LOG"
