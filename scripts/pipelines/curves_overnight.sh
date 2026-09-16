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
    # Already present AND clean? keep it. Duplicated? regenerate it.
    if '"$PY"' scripts/verify_pools.py --glob "$out" >/dev/null 2>&1; then
      echo "SKIP clean: $out"; exit 0
    fi
    echo "REGENERATE (failed verification): $out"; rm -f "$out" "${out%.npz}__labeled.npz"
  fi
  eval "'"$PY"' -m albench.cli ${line#albench }"')
echo "  gen        $J1  (${NGEN} pools)"

J2=$(sub verify_pools "$J1" "$CPU --cpus-per-task=2 --mem=32G --time=01:00:00" \
     "$PY scripts/verify_pools.py --glob 'outputs/curves/pools/*.npz'")
echo "  verify     $J2"

J3=$(sub label "$J2" "$GPU --array=1-24%8" '
  f=$(ls outputs/curves/pools/*.npz 2>/dev/null | grep -v labeled | sed -n "${ALBENCH_TASK_ID}p")
  [ -z "$f" ] && exit 0
  out="${f%.npz}__labeled.npz"; [ -s "$out" ] && { echo "SKIP $out"; exit 0; }
  '"$PY"' scripts/label_pool.py --in "$f" --out "$out"')
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
J6=$(sub train "$J5" "--partition=${ALBENCH_GPU_PARTITION:-gpuq} --qos=slow_nice --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=11:30:00 --array=1-${NTR}%20" '
  line=$(sed -n "${ALBENCH_TASK_ID}p" outputs/curves/jobs.txt); [ -z "$line" ] && exit 0
  out=$(echo "$line" | sed -n "s/.*--output-dir \([^ ]*\).*/\1/p")
  if [ -n "$(find "$out" -name result.json -size +0c -print -quit 2>/dev/null)" ]; then
    echo "SKIP $out"; exit 0; fi
  eval "'"$PY"' $line"')
echo "  train      $J6  (${NTR} points)"

J7=$($SB --parsable --job-name=cv_wd --dependency=after:"$J1" \
     --partition="${ALBENCH_CPU_PARTITION:-cpuq}" --qos=slow_nice --cpus-per-task=1 --mem=2G \
     --time=24:00:00 --output="$REPO/logs/curves/watchdog-%A.out" --chdir="$REPO" \
     --wrap="bash scripts/pipelines/curves_watchdog.sh")
echo "  watchdog   $J7"
echo
echo "chain: $J1 -> $J2 -> $J3 -> $J4 -> $J5 -> $J6   (watchdog $J7)"
