#!/bin/bash
# Watchdog for the canonical scaling-curve HP-search (outputs/scaling_canonical_oct13).
# - Holds ALL cells while the laptop GitHub-bridge heartbeat is stale.
# - LEAST-COVERED-FIRST ordering -> even coverage across reservoirs x D.
# - DYNAMIC QOS: each submitted cell is placed on the highest-priority qos with a free
#   slot for its GPU type (fast -> default -> slow_nice), so every non-preemptible slot
#   (fast 2H+2V, default 4H+4V) stays full and freed GPUs go to the neediest cell on a
#   qos that actually wins them. slow_nice (and fast) use a SHORT 4h walltime so they can
#   backfill onto momentarily-free cluster GPUs; default keeps 12h. qos/time are passed
#   as sbatch CLI overrides, so the per-cell sbatch files are left untouched.
# - Done = >=TARGET usable models (metas with no "error" field).
set -u
R=/grid/wsbs/home_norepl/christen/ALBench-S2F
SB=/cm/shared/apps/slurm/current/bin
BASE=$R/outputs/scaling_canonical_oct13
TARGET=40
CAP_MAX=18        # total in-queue ceiling (run+pend); keeps non-preempt full + backfill buffer
CAP_MIN=10
CAP=$CAP_MAX
RATE_LO=800
RATE_HI=2000
# per-user qos GPU caps (admin-set): fast 2, default 4 for each of h100/v100
FAST_CAP=2
DEF_CAP=4
cd "$R"

cell_count() { grep -L '"error"' "$1"/r*_meta.json 2>/dev/null | wc -l; }

while true; do
  echo "=== sc sweep $(date -u +%FT%TZ) ==="
  PROBE=$(PYTHONPATH=$R uv run --no-sync python -c "
import sys; sys.path.insert(0,'$R')
import albench.arch_search.gh_bridge as g
a = g.bridge_age_s(); r = g.rate_state().get('remaining')
print(f\"{int(a) if a is not None else 99999}|{r if r is not None else -1}\")
" 2>/dev/null || echo "99999|-1")
  BRIDGE_AGE=${PROBE%%|*}
  REMAIN=${PROBE##*|}
  if [ "${BRIDGE_AGE:-99999}" -gt 420 ]; then
    echo "  bridge heartbeat ${BRIDGE_AGE}s old -> laptop asleep; all cells held"
    sleep 900; continue
  fi
  echo "  bridge heartbeat ${BRIDGE_AGE}s old -> alive (budget remaining=${REMAIN})"

  RL=$(find "$BASE/logs" -name 'sc_*.log' -mmin -8 2>/dev/null -print0 \
       | xargs -0 grep -lE "RateLimitExceeded|usage/rate limit|rate limit|overloaded" 2>/dev/null | wc -l)
  if { [ "${REMAIN}" -ge 0 ] && [ "${REMAIN}" -lt "$RATE_LO" ]; } || [ "${RL:-0}" -ge 1 ]; then
    NEW=$(( CAP - 2 )); [ "$NEW" -lt "$CAP_MIN" ] && NEW=$CAP_MIN
    [ "$NEW" != "$CAP" ] && echo "  rate pressure (remaining=${REMAIN}, rl_logs=${RL}) -> cap ${CAP}->${NEW}"
    CAP=$NEW
  elif [ "${REMAIN}" -ge "$RATE_HI" ]; then
    NEW=$(( CAP + 1 )); [ "$NEW" -gt "$CAP_MAX" ] && NEW=$CAP_MAX
    [ "$NEW" != "$CAP" ] && echo "  budget healthy (remaining=${REMAIN}) -> cap ${CAP}->${NEW}"
    CAP=$NEW
  fi

  running_all=$($SB/squeue -u christen -h -o "%j" 2>/dev/null)         # run + pend (dedup)
  nq=$(echo "$running_all" | grep -cE "^sc_" || true)                  # total in queue
  echo "  sc cells in queue: $nq / cap $CAP"

  # free non-preempt slots per (qos,gpu), from RUNNING jobs only
  RJ=$($SB/squeue -u christen -t RUNNING -h -o "%q|%b" 2>/dev/null)
  cnt() { echo "$RJ" | grep -cE "$1"; }
  ffh=$(( FAST_CAP - $(cnt '^fast\|.*h100') )); ffv=$(( FAST_CAP - $(cnt '^fast\|.*v100') ))
  fdh=$(( DEF_CAP  - $(cnt '^default\|.*h100') )); fdv=$(( DEF_CAP - $(cnt '^default\|.*v100') ))
  echo "  free non-preempt slots  H100: fast=$ffh default=$fdh  |  V100: fast=$ffv default=$fdv"

  # least-covered-first order
  mapfile -t ORDER < <(
    for f in "$BASE"/sc_*.sbatch; do
      job=$(basename "$f" .sbatch)
      [ "$job" = "sc_watchdog" ] && continue
      case "$job" in sc_smoke_*) continue;; esac
      rest=${job#sc_}; D=${rest##*_}; RES=${rest%_*}
      printf '%06d\t%s\n' "$(cell_count "$BASE/${RES}_d${D}")" "$f"
    done | sort -n -k1,1 | cut -f2-
  )

  for f in "${ORDER[@]}"; do
    job=$(basename "$f" .sbatch)
    echo "$running_all" | grep -qE "^${job}[vw]?$" && continue
    rest=${job#sc_}; D=${rest##*_}; RES=${rest%_*}
    n=$(cell_count "$BASE/${RES}_d${D}")
    [ "$n" -ge "$TARGET" ] && continue
    if [ "$nq" -ge "$CAP" ]; then
      echo "  at cap ($nq); defer $job ($n/$TARGET)"; continue
    fi
    # GPU type from the cell's own gres; qos = highest-priority free slot for it
    g=$(grep -oE 'gpu:(h100|v100)' "$f" | head -1); g=${g##*:}
    if [ "$g" = h100 ]; then
      if   [ "$ffh" -gt 0 ]; then Q=fast;      T=4:00:00;     ffh=$((ffh-1))
      elif [ "$fdh" -gt 0 ]; then Q=default;   T=12:00:00;    fdh=$((fdh-1))
      else                        Q=slow_nice; T=4:00:00; fi
    else
      if   [ "$ffv" -gt 0 ]; then Q=fast;      T=4:00:00;     ffv=$((ffv-1))
      elif [ "$fdv" -gt 0 ]; then Q=default;   T=12:00:00;    fdv=$((fdv-1))
      else                        Q=slow_nice; T=4:00:00; fi
    fi
    if $SB/sbatch --qos="$Q" --time="$T" "$f" >/dev/null 2>&1; then
      nq=$((nq+1)); echo "  submitted $job ($n/$TARGET) qos=$Q[$g] -> queue $nq"
    fi
  done
  sleep 420
done
