#!/bin/bash
# Submit the whole curve pipeline as one dependency chain, with verification gates.
#
# generate -> VERIFY -> label -> VERIFY(labels) -> link -> train -> watchdog
#
# The gates are the point. Labelling costs GPU-hours and training costs many more, and
# a duplicated or unlabelled pool silently poisons every curve point built from it. A
# gate that exits non-zero stops the chain there (afterok), so a bad pool costs minutes
# instead of a night.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
[ -f scripts/cluster/site.env ] && . scripts/cluster/site.env
SB="${ALBENCH_SBATCH:-$(command -v sbatch)}"
PY="${ALBENCH_PYTHON:-uv run --no-sync python}"
mkdir -p logs/curves outputs/curves/pools

sub () {  # sub <name> <dep|none> <extra-sbatch-args> <command>
  local name="$1" dep="$2" extra="$3" cmd="$4" d=""
  [ "$dep" != none ] && d="--dependency=afterok:$dep"
  # shellcheck disable=SC2086
  $SB --parsable --job-name="cv_$name" $d $extra \
      --output="$REPO/logs/curves/${name}-%A_%a.out" --chdir="$REPO" \
      --wrap="export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 ALBENCH_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}; $cmd"
}

CPU="--partition=${ALBENCH_CPU_PARTITION:-cpuq} --qos=slow_nice --cpus-per-task=8 --mem=96G --time=11:30:00"
GPU="--partition=${ALBENCH_GPU_PARTITION:-gpuq} --qos=slow_nice --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=11:30:00"

# 1. regenerate anything missing or duplicated (dedupe is on by default now)
$PY scripts/build_curve_plan.py --stage generate >/dev/null
NGEN=$(wc -l < outputs/curves/generate.txt)
J1=$(sub gen none "$CPU --array=1-${NGEN}%10" '
  line=$(sed -n "${ALBENCH_TASK_ID}p" outputs/curves/generate.txt); [ -z "$line" ] && exit 0
  out=$(echo "$line" | sed -n "s/.*--out \([^ ]*\).*/\1/p")
  if [ -s "$out" ]; then
    # Reuse only if it is BOTH clean and generated under the parameters the plan asks
    # for now. A config edit otherwise leaves stale pools that are silently reused, so
    # the curve gets labelled with parameters its data was not generated under.
    strat=$(echo "$line" | sed -n "s/.*--strategy \([^ ]*\).*/\1/p")
    sets=$(echo "$line" | grep -oE -- "--set [^ ]+" | sed "s/--set /--set /" | tr "\n" " ")
    if '"$PY"' scripts/verify_pools.py --glob "$out" >/dev/null 2>&1 &&
       '"$PY"' scripts/pool_params_match.py --pool "$out" --strategy "$strat" $sets; then
      echo "SKIP clean and current: $out"; exit 0
    fi
    echo "REGENERATE (stale or failed verification): $out"; rm -f "$out" "${out%.npz}__labeled.npz"
  fi
  eval "'"$PY"' -m albench.cli ${line#albench }"')
echo "  gen        $J1  (${NGEN} pools)"

J2=$(sub verify_pools "$J1" "$CPU --cpus-per-task=2 --mem=32G --time=01:00:00" \
     "$PY scripts/verify_pools.py --glob 'outputs/curves/pools/*.npz'")
echo "  verify     $J2"

# Each task CLAIMS a pool rather than indexing into a live `ls`. Index-based mapping
# renumbers the moment a pool is added or removed -- which happened: dropping a stale
# pool mid-flight shifted every later index, so a running task was labelling a
# different file than its index implied. Claiming is immune to that, and lets the
# array be sized generously without caring about the exact pool count.
J3=$(sub label "$J2" "$GPU --array=1-16%8" '
  for f in outputs/curves/pools/*.npz; do
    case "$f" in *__labeled.npz) continue;; esac
    out="${f%.npz}__labeled.npz"
    [ -s "$out" ] && continue
    mkdir -p outputs/curves/.labelclaims
    claim="outputs/curves/.labelclaims/$(basename "${f%.npz}")"
    if ! mkdir "$claim" 2>/dev/null; then
      if [ -n "$(find "$claim" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then
        rmdir "$claim" 2>/dev/null; mkdir "$claim" 2>/dev/null || continue
      else continue; fi
    fi
    echo "labelling $f"
    '"$PY"' scripts/label_pool.py --in "$f" --out "$out"; rc=$?
    rmdir "$claim" 2>/dev/null
    [ $rc -ne 0 ] && { echo "FAILED $f"; exit $rc; }
  done
  echo "label task done"')
echo "  label      $J3"

J4=$(sub verify_labels "$J3" "$CPU --cpus-per-task=2 --mem=32G --time=01:00:00" \
     "$PY scripts/verify_pools.py --glob 'outputs/curves/pools/*.npz' --require-labels")
echo "  verify_lab $J4"

J5=$(sub link "$J4" "$CPU --cpus-per-task=2 --mem=32G --time=01:00:00" \
     "$PY scripts/link_screen_pools.py --cache outputs/curves/pools --out outputs/curves/pools_linked --glob '*__labeled.npz' && $PY scripts/build_curve_plan.py --stage train")
echo "  link+plan  $J5"

# Train count is not known until the plan is rebuilt, so size the array from the
# current plan and let the watchdog resubmit if the plan grew.
$PY scripts/build_curve_plan.py --stage train >/dev/null
NTR=$(wc -l < outputs/curves/jobs.txt)
# Train on ALL THREE tiers at once. They are independent allocations, so one array
# per tier gets roughly 28 concurrent GPUs instead of slow_nice's 20 -- but three
# arrays over one job list means two tasks can reach for the same curve point, so
# each task must CLAIM its point atomically first.
#
# mkdir is the claim primitive on purpose: it is atomic on POSIX and, unlike flock,
# behaves on NFS. A claim with no result that is older than 3h is stolen, so a task
# killed by preemption (SIGKILL skips the trap) does not strand its point forever.
TRAIN_BODY='
  line=$(sed -n "${ALBENCH_TASK_ID}p" outputs/curves/jobs.txt); [ -z "$line" ] && exit 0
  out=$(echo "$line" | sed -n "s/.*--output-dir \([^ ]*\).*/\1/p")
  if [ -n "$(find "$out" -name result.json -size +0c -print -quit 2>/dev/null)" ]; then
    echo "SKIP done: $out"; exit 0; fi
  mkdir -p "$out"
  if ! mkdir "$out/.claim" 2>/dev/null; then
    if [ -n "$(find "$out/.claim" -maxdepth 0 -mmin +180 2>/dev/null)" ]; then
      echo "stealing stale claim: $out"; rmdir "$out/.claim" 2>/dev/null
      mkdir "$out/.claim" 2>/dev/null || { echo "SKIP claimed: $out"; exit 0; }
    else
      echo "SKIP claimed by another task: $out"; exit 0
    fi
  fi
  trap "rmdir \"$out/.claim\" 2>/dev/null" EXIT
  eval "'"$PY"' $line"'

GPUBASE="--partition=${ALBENCH_GPU_PARTITION:-gpuq} --gres=gpu:1 --cpus-per-task=8 --mem=64G"
J6=$(sub train "$J5" "$GPUBASE --qos=slow_nice --time=11:30:00 --array=1-${NTR}%20" "$TRAIN_BODY")
J6b=$(sub train_fast "$J5" "$GPUBASE --qos=fast --time=03:30:00 --array=1-${NTR}%2" "$TRAIN_BODY")
J6c=$(sub train_def "$J5" "$GPUBASE --qos=default --time=11:30:00 --array=1-${NTR}%4" "$TRAIN_BODY")
echo "  train      $J6 (slow_nice %20) $J6b (fast %2) $J6c (default %4)  -- ${NTR} points"

J7=$($SB --parsable --job-name=cv_wd --dependency=after:"$J1" \
     --partition="${ALBENCH_CPU_PARTITION:-cpuq}" --qos=slow_nice --cpus-per-task=1 --mem=2G \
     --time=24:00:00 --output="$REPO/logs/curves/watchdog-%A.out" --chdir="$REPO" \
     --wrap="bash scripts/pipelines/curves_watchdog.sh")
echo "  watchdog   $J7"
echo
echo "chain: $J1 -> $J2 -> $J3 -> $J4 -> $J5 -> $J6   (watchdog $J7)"
