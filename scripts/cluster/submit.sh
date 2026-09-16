#!/bin/bash
# Submit one job (or a job array) without hardcoding anyone's cluster.
#
# WHY THIS EXISTS. Every runner in this repo used to be a full sbatch file carrying
# one site's partition names, QoS names, absolute sbatch path and absolute repo path.
# That makes the science unreproducible anywhere else: a reader cannot tell which
# lines are the experiment and which are local scheduling trivia. Here the experiment
# is the command after `--`, and everything else comes from scripts/cluster/site.env.
#
# It also degrades honestly: with no scheduler present it runs the command in the
# foreground, so the pipelines work on a laptop or inside an interactive allocation.
#
# Usage:
#   scripts/cluster/submit.sh --name screen --gpus 1 --cpus 8 --mem 64G \
#       --time 11:30:00 --array 1-105%20 -- python -m albench.cli doctor
#
# Options: --name --gpus --cpus --mem --time --array --tier --constraint --cpu-only
#          --tiered   spread across ALBENCH_TIERS, asking the scheduler where each starts
# Anything after `--` is the command. $ALBENCH_TASK_ID is set for array members
# (the scheduler's index, or the loop index in local mode).
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
[ -f "$REPO_DIR/scripts/cluster/site.env" ] && . "$REPO_DIR/scripts/cluster/site.env"
[ -n "${ALBENCH_SITE_ENV:-}" ] && [ -f "$ALBENCH_SITE_ENV" ] && . "$ALBENCH_SITE_ENV"

NAME=job; GPUS=0; CPUS=4; MEM=16G; TIME=01:00:00; ARRAY=""; TIER=""; CONSTRAINT=""; CPU_ONLY=0
TIERED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --array) ARRAY="$2"; shift 2;;
    --tier) TIER="$2"; shift 2;;
    --constraint) CONSTRAINT="$2"; shift 2;;
    --cpu-only) CPU_ONLY=1; shift;;
    --tiered) TIERED=1; shift;;
    --) shift; break;;
    *) echo "submit.sh: unknown option $1" >&2; exit 2;;
  esac
done
[ $# -eq 0 ] && { echo "submit.sh: no command given (everything after -- is the command)" >&2; exit 2; }

SBATCH="${ALBENCH_SBATCH:-$(command -v sbatch 2>/dev/null)}"
BACKEND="${ALBENCH_BACKEND:-}"
if [ -z "$BACKEND" ]; then
  if [ -n "$SBATCH" ] && [ -x "$SBATCH" ]; then BACKEND=slurm; else BACKEND=local; fi
fi
RUN_DIR="${ALBENCH_REPO:-$REPO_DIR}"

if [ "$BACKEND" = local ]; then
  # No scheduler: run it here. An array becomes a sequential loop, which keeps the
  # pipelines runnable (slowly) rather than silently doing nothing.
  echo "[submit] local backend: running '$NAME' in the foreground"
  cd "$RUN_DIR" || exit 1
  if [ -n "$ARRAY" ]; then
    lo=${ARRAY%%-*}; rest=${ARRAY#*-}; hi=${rest%%[%:]*}
    for i in $(seq "$lo" "$hi"); do
      echo "[submit] $NAME task $i"
      ALBENCH_TASK_ID=$i "$@" || echo "[submit] task $i failed (continuing)"
    done
  else
    ALBENCH_TASK_ID=0 "$@"
  fi
  exit $?
fi

mkdir -p "$RUN_DIR/logs/$NAME"
args=(--job-name="$NAME" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME"
      --output="$RUN_DIR/logs/$NAME/%A_%a.out" --chdir="$RUN_DIR")
if [ "$CPU_ONLY" -eq 1 ] || [ "$GPUS" -eq 0 ]; then
  [ -n "${ALBENCH_CPU_PARTITION:-}" ] && args+=(--partition="$ALBENCH_CPU_PARTITION")
else
  [ -n "${ALBENCH_GPU_PARTITION:-}" ] && args+=(--partition="$ALBENCH_GPU_PARTITION")
  args+=(--gres=gpu:"$GPUS")
  c="${CONSTRAINT:-${ALBENCH_GPU_CONSTRAINT:-}}"
  [ -n "$c" ] && args+=(--constraint="$c")
fi
[ -n "${ALBENCH_ACCOUNT:-}" ] && args+=(--account="$ALBENCH_ACCOUNT")
[ -n "$TIER" ] && args+=(--qos="$TIER")
[ -n "$ARRAY" ] && args+=(--array="$ARRAY")

# TQDM_DISABLE keeps progress bars out of the logs: they make a multi-megabyte file
# out of a job whose real output is twenty lines.
WRAP="export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 ALBENCH_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}; $(printf '%q ' "$@")"

if [ "$TIERED" -eq 1 ] && [ -z "$TIER" ] && [ -n "${ALBENCH_TIERS:-}" ]; then
  # Pick a QoS tier by ASKING the scheduler where the job would actually start, rather
  # than tracking capacity ourselves. A hardcoded per-tier cap is invisible when wrong:
  # the job is accepted and then sits pending while another tier has room.
  SQ="${ALBENCH_SQUEUE:-$(command -v squeue 2>/dev/null)}"
  [ -z "$SQ" ] && SQ="$(dirname "$SBATCH")/squeue"
  MINE=$($SQ -u "$USER" -h -o "%A" 2>/dev/null | tr '\n' ' ')
  read -r -a _tiers <<< "$ALBENCH_TIERS"
  now=$(date +%s); best_q=""; best_w=""; best_d=""
  for t in "${_tiers[@]}"; do
    q="${t%%:*}"; w="${t#*:}"; [ "$w" = "$q" ] && w="$TIME"
    probe=$("$SBATCH" --test-only --qos="$q" --time="$w" "${args[@]}" --wrap="true" 2>&1)
    ts=$(printf '%s' "$probe" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
    [ -z "$ts" ] && continue
    # A tier can report "starts now" only because it would PREEMPT something, and on a
    # busy cluster that something is usually our own lower-tier job. Trading a running
    # job for a queued one is a net loss, so that placement does not count as free.
    victims=$(printf '%s' "$probe" | sed -n 's/.*Preempts: *//p' | tr ',' ' ')
    skip=0
    for v in $victims; do
      case " $MINE " in *" ${v%%_*} "*) skip=1;; esac
    done
    [ "$skip" -eq 1 ] && continue
    st=$(date -d "$ts" +%s 2>/dev/null) || continue
    d=$(( st - now )); [ "$d" -lt 0 ] && d=0
    if [ -z "$best_d" ] || [ "$d" -lt "$best_d" ]; then best_q=$q; best_w=$w; best_d=$d; fi
    [ "$d" -le 300 ] && break
  done
  if [ -n "$best_q" ]; then
    echo "[submit] tier '$best_q' starts in ~$((best_d/60)) min"
    args+=(--qos="$best_q" --time="$best_w")
  fi
fi

"$SBATCH" "${args[@]}" --wrap="$WRAP"
