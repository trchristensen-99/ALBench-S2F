#!/bin/bash
# Watchdog for the canonical scaling-curve HP-search (outputs/scaling_canonical_oct13).
# - Holds ALL cells while the laptop GitHub-bridge heartbeat is stale (resubmitting
#   then just burns a job that reads a stale heartbeat and exits with LLM_FAIL_HARD).
# - ADAPTIVE concurrency cap: targets CAP_MAX (16) so H100+V100 fill together, but
#   backs off toward CAP_MIN (10) when the GitHub-bridge budget runs thin or a cell
#   logs a rate-limit, then recovers (step +1) once the budget refills. The single
#   poller serialises proposals and each cell only needs one every few minutes, so
#   16 concurrent cells is comfortable until the per-user GitHub quota says otherwise.
# - A cell is DONE at >=TARGET USABLE models: metas WITHOUT an "error" field (a crashed
#   model still writes a meta, so counting all metas would falsely look done).
# - Cells are processed LEAST-COVERED-FIRST so coverage balances across all reservoirs
#   (alphabetical order starved genomic/motif/random/zoonomia/gc_matched while it drove
#   the alphabetically-early arms to completion). This gives every reservoir a
#   preliminary point at every D before any single cell is driven to saturation.
# - Each cell resubmits to its OWN sbatch so its qos/walltime/gres is preserved; the
#   driver resumes and skips models already present.
set -u
R=/grid/wsbs/home_norepl/christen/ALBench-S2F
SB=/cm/shared/apps/slurm/current/bin
BASE=$R/outputs/scaling_canonical_oct13
TARGET=40
CAP_MAX=16        # concurrency ceiling (fills H100 + V100 together)
CAP_MIN=10        # floor under bridge rate-limit pressure
CAP=$CAP_MAX      # current adaptive cap (persists across sweeps)
RATE_LO=800       # GitHub-bridge budget below this -> back off toward CAP_MIN
RATE_HI=2000      # budget above this -> recover toward CAP_MAX
cd "$R"

# usable-model count for a cell out-dir (metas with no "error" field)
cell_count() { grep -L '"error"' "$1"/r*_meta.json 2>/dev/null | wc -l; }

while true; do
  echo "=== sc sweep $(date -u +%FT%TZ) ==="
  # One probe returns BOTH the heartbeat age and the live GitHub-bridge budget
  # (bridge_age_s makes an API read, which populates the rate state).
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

  # --- adaptive cap: back off on a thin budget OR a recent cell rate-limit log ---
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

  running=$($SB/squeue -u christen -h -o "%j" 2>/dev/null)
  nrun=$(echo "$running" | grep -cE "^sc_" || true)
  echo "  running sc cells: $nrun / cap $CAP"

  # Build the sweep order: least-covered cell first (balances reservoirs).
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
    # already running (allow a trailing qos/variant char)?
    echo "$running" | grep -qE "^${job}[vw]?$" && continue
    rest=${job#sc_}
    D=${rest##*_}
    RES=${rest%_*}
    out="$BASE/${RES}_d${D}"
    n=$(cell_count "$out")
    [ "$n" -ge "$TARGET" ] && continue
    if [ "$nrun" -ge "$CAP" ]; then
      echo "  at cap ($nrun); defer $job ($n/$TARGET)"
      continue
    fi
    $SB/sbatch "$f" >/dev/null 2>&1 && { nrun=$((nrun+1)); echo "  submitted $job ($n/$TARGET) -> running $nrun"; }
  done
  sleep 420
done
