#!/bin/bash
# Spread single-fold GPU jobs across every QoS tier we can use, H100 only.
#
# WHY TIER AT ALL. The three tiers are INDEPENDENT allocations, so submitting
# everything to one leaves the others idle. Measured 2026-09-15:
#
#   qos         priority  maxwall   starts
#   fast            4000       4h   immediately
#   default         1000      12h   immediately
#   slow_nice        100      30d   only when H100s are free cluster-wide
#
# bio_ai and cryo are visible in sacctmgr but rejected for this account.
#
# WHY --constraint=h100 IS NOT OPTIONAL FOR LSTM WORK. DREAM-RNN epochs measured
# ~5 min on H100 against ~38 min on V100, a 7.6x penalty; LegNet (a CNN) pays only
# 2.4x, so LegNet jobs are better off left unconstrained where idle V100s are
# plentiful. Without the constraint Slurm packed four yeast-oracle folds onto one
# V100 node and they were still on epoch 1 of 6 an hour in.
#
# WHY WE ASK SLURM INSTEAD OF HARDCODING CAPS. A previous version carried a table
# of per-tier GPU caps. The table was wrong -- it claimed fast held 2 concurrent
# H100 jobs when it actually holds 4 -- and a wrong cap is invisible: the job is
# accepted and then sits PENDING with QOSMaxGRESPerUser while another tier has room.
# `sbatch --test-only` reports the time a job WOULD start, accounting for every
# limit that applies, so we place each job in the highest-priority tier that starts
# it now and fall back to whichever tier starts it soonest.
#
# Usage:
#   scripts/slurm/launch_tiered.sh <runner.sbatch> <extra-args> -- <id> [<id> ...]
# Example:
#   scripts/slurm/launch_tiered.sh run_yofold_one.sbatch "--tag _B" -- 0 1 2 3
set -uo pipefail
SB=/cm/shared/apps/slurm/current/bin/sbatch

RUNNER="${1:?usage: launch_tiered.sh <runner.sbatch> <extra-args> -- <ids...>}"
EXTRA="${2:-}"
shift 2
[ "${1:-}" = "--" ] && shift

# tier:walltime, highest priority first
TIERS=("fast:03:30:00" "default:11:30:00" "slow_nice:11:30:00")
SOON=300   # seconds; a job starting within this counts as "starts now"

# Job IDs we already own, so a tier that would preempt them can be recognised.
MYJOBS=$(/cm/shared/apps/slurm/current/bin/squeue -u "$USER" -h -o "%A" 2>/dev/null | tr '\n' ' ')

for id in "$@"; do
  MYJOBS="$MYJOBS $(/cm/shared/apps/slurm/current/bin/squeue -u "$USER" -h -o "%A" 2>/dev/null | tr '\n' ' ')"
  best_qos=""; best_wall=""; best_delay=""
  now=$(date +%s)
  for t in "${TIERS[@]}"; do
    qos="${t%%:*}"; wall="${t#*:}"
    out=$($SB --test-only --qos="$qos" --time="$wall" --constraint=h100 \
              --export=ALL,FOLD_ID="$id",EXTRA_ARGS="$EXTRA" "$RUNNER" 2>&1)
    ts=$(printf '%s' "$out" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
    # No timestamp means the tier rejected the job outright (qos not allowed, walltime
    # over the tier max). Skip it rather than treating it as infinitely slow.
    [ -z "$ts" ] && continue
    # A high-priority tier can report "starts now" only because it would PREEMPT
    # something -- and on a busy cluster the thing it preempts is usually our own
    # lower-tier job. Trading a running job for a queued one is a net loss, so a
    # placement that evicts our own work does not count as available.
    victims=$(printf '%s' "$out" | sed -n 's/.*Preempts: *//p' | tr ',' ' ')
    if [ -n "$victims" ]; then
      mine=0
      for v in $victims; do
        case " $MYJOBS " in *" ${v%%_*} "*) mine=1 ;; esac
      done
      [ "$mine" -eq 1 ] && continue
    fi
    start=$(date -d "$ts" +%s 2>/dev/null) || continue
    delay=$(( start - now )); [ "$delay" -lt 0 ] && delay=0
    if [ "$delay" -le "$SOON" ]; then best_qos=$qos; best_wall=$wall; best_delay=$delay; break; fi
    if [ -z "$best_delay" ] || [ "$delay" -lt "$best_delay" ]; then
      best_qos=$qos; best_wall=$wall; best_delay=$delay
    fi
  done

  if [ -z "$best_qos" ]; then
    printf '  fold %-3s -> NO TIER ACCEPTED IT\n' "$id"
    continue
  fi
  out=$($SB --qos="$best_qos" --time="$best_wall" --constraint=h100 \
            --export=ALL,FOLD_ID="$id",EXTRA_ARGS="$EXTRA" "$RUNNER" 2>&1)
  if [[ "$out" == *"Submitted"* ]]; then
    if [ "$best_delay" -le "$SOON" ]; then when="starts now"; else when="starts in ~$((best_delay/60)) min"; fi
    printf '  fold %-3s -> %-10s %-9s (%s)\n' "$id" "$best_qos" "${out##* }" "$when"
  else
    printf '  fold %-3s -> %-10s FAILED: %s\n' "$id" "$best_qos" "$(printf '%s' "$out" | tail -1)"
  fi
done
