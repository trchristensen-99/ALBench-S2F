#!/bin/bash
# Watchdog for the canonical scaling-curve HP-search (outputs/scaling_canonical_oct13).
# - Holds ALL cells while the laptop GitHub-bridge heartbeat is stale (resubmitting
#   then just burns a job that reads a stale heartbeat and exits with LLM_FAIL_HARD).
# - Enforces a global concurrency cap (bridge rate limit ~10 concurrent LLM cells).
# - A cell is DONE at >=60 USABLE models: metas WITHOUT an "error" field (a crashed
#   model still writes a meta, so counting all metas would falsely look done).
# - Each cell resubmits to its OWN sbatch so its qos/walltime is preserved; the
#   driver resumes and skips models already present.
set -u
R=/grid/wsbs/home_norepl/christen/ALBench-S2F
SB=/cm/shared/apps/slurm/current/bin
BASE=$R/outputs/scaling_canonical_oct13
TARGET=40
CAP=10            # max concurrent scaling cells (bridge rate-limit budget)
cd "$R"

while true; do
  echo "=== sc sweep $(date -u +%FT%TZ) ==="
  BRIDGE_AGE=$(PYTHONPATH=$R uv run --no-sync python -c "
import sys; sys.path.insert(0,'$R')
from albench.arch_search.gh_bridge import bridge_age_s
a = bridge_age_s(); print(int(a) if a is not None else 99999)
" 2>/dev/null || echo 99999)
  if [ "${BRIDGE_AGE:-99999}" -gt 420 ]; then
    echo "  bridge heartbeat ${BRIDGE_AGE}s old -> laptop asleep; all cells held"
    sleep 900; continue
  fi
  echo "  bridge heartbeat ${BRIDGE_AGE}s old -> alive"

  running=$($SB/squeue -u christen -h -o "%j" 2>/dev/null)
  nrun=$(echo "$running" | grep -cE "^sc_" || true)
  echo "  running sc cells: $nrun / cap $CAP"

  for f in "$BASE"/sc_*.sbatch; do
    job=$(basename "$f" .sbatch)
    [ "$job" = "sc_watchdog" ] && continue
    case "$job" in sc_smoke_*) continue;; esac
    # already running (allow a trailing qos/variant char)?
    echo "$running" | grep -qE "^${job}[vw]?$" && continue
    # derive out_dir: sc_<res>_<D> -> <res>_d<D>
    rest=${job#sc_}
    D=${rest##*_}
    RES=${rest%_*}
    out="$BASE/${RES}_d${D}"
    n=$(grep -L '"error"' "$out"/r*_meta.json 2>/dev/null | wc -l)
    [ "$n" -ge "$TARGET" ] && continue
    if [ "$nrun" -ge "$CAP" ]; then
      echo "  at cap ($nrun); defer $job ($n/$TARGET)"
      continue
    fi
    $SB/sbatch "$f" >/dev/null 2>&1 && { nrun=$((nrun+1)); echo "  submitted $job ($n/$TARGET) -> running $nrun"; }
  done
  sleep 420
done
