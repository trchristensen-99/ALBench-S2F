#!/bin/bash
# Spread single-fold GPU jobs across every QoS tier we can use, H100 only.
#
# WHY. The three tiers are INDEPENDENT allocations, so submitting everything to one
# of them leaves the others idle. Measured 2026-09-15:
#
#   qos         priority  maxwall   H100 cap   starts
#   fast            4000       4h          2   immediately
#   default         1000      12h          4   immediately
#   slow_nice        100      30d         20   queued ~2h out
#
# That is 26 concurrent H100s if all three are used, against 20 from slow_nice alone
# -- and the first 6 start now rather than in two hours. bio_ai and cryo are visible
# in sacctmgr but rejected for our account.
#
# --constraint=h100 is not optional for LSTM work. DREAM-RNN epochs measured ~5 min on
# H100 and ~38 min on V100, a 7.6x penalty (LegNet, a CNN, only pays 2.4x). Without
# the constraint Slurm will place jobs on V100s and pack several per node.
#
# Usage:
#   scripts/slurm/launch_tiered.sh <runner.sbatch> <extra-args> -- <id> [<id> ...]
# Example:
#   scripts/slurm/launch_tiered.sh run_yoracleB_one.sbatch "--tag _B" -- 0 1 2 3
set -uo pipefail
SB=/cm/shared/apps/slurm/current/bin/sbatch

RUNNER="${1:?usage: launch_tiered.sh <runner.sbatch> <extra-args> -- <ids...>}"
EXTRA="${2:-}"
shift 2
[ "${1:-}" = "--" ] && shift

# tier:capacity:walltime -- fast first so the highest-priority slots fill first
TIERS=("fast:2:03:30:00" "default:4:11:30:00" "slow_nice:20:11:30:00")

# Seed the per-tier counters from jobs ALREADY queued, not from zero: each
# invocation otherwise re-fills a tier that a previous call had already filled, and
# the excess sits pending instead of running somewhere with free capacity.
declare -A used
for t in "${TIERS[@]}"; do
  qos="${t%%:*}"
  used[$qos]=$(/cm/shared/apps/slurm/current/bin/squeue -u "$USER" -h -o "%q" 2>/dev/null \
               | grep -c "^${qos}$")
done
echo "already queued per tier: $(for k in "${!used[@]}"; do printf "%s=%s " "$k" "${used[$k]}"; done)"
i=0
for id in "$@"; do
  placed=0
  for t in "${TIERS[@]}"; do
    qos="${t%%:*}"; rest="${t#*:}"; cap="${rest%%:*}"; wall="${rest#*:}"
    n=${used[$qos]:-0}
    if [ "$n" -lt "$cap" ]; then
      out=$($SB --qos="$qos" --time="$wall" --constraint=h100 \
                --export=ALL,FOLD_ID="$id",EXTRA_ARGS="$EXTRA" "$RUNNER" 2>&1)
      if [[ "$out" == *"Submitted"* ]]; then
        printf "  fold %-3s -> %-10s %s\n" "$id" "$qos" "${out##* }"
        used[$qos]=$((n+1)); placed=1; break
      else
        printf "  fold %-3s -> %-10s REJECTED: %s\n" "$id" "$qos" "$(echo "$out"|tail -1)"
        used[$qos]=$cap   # tier is full or unavailable; stop trying it
      fi
    fi
  done
  [ "$placed" -eq 0 ] && printf "  fold %-3s -> NO TIER HAD CAPACITY (submit it later)\n" "$id"
  i=$((i+1))
done
echo "placed across tiers: ${used[*]:-none}"
