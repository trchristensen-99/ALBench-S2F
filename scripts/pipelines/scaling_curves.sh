#!/bin/bash
# Additive scaling curves: fixed genomic baseline + N sequences from one reservoir.
#
#   ./scripts/pipelines/scaling_curves.sh plan       # what would run, and the cost
#   ./scripts/pipelines/scaling_curves.sh generate   # one 300k pool per reservoir
#   ./scripts/pipelines/scaling_curves.sh label      # oracle-label those pools
#   ./scripts/pipelines/scaling_curves.sh link       # expose pools to the driver
#   ./scripts/pipelines/scaling_curves.sh train      # every curve point
#   ./scripts/pipelines/scaling_curves.sh status
#
# ONE labelled 300k pool per reservoir serves every point on its curve: increments
# are nested prefixes of a single permutation, so no point requires its own pool and
# subset-order replicates are free.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
SUBMIT=scripts/cluster/submit.sh
PY="${ALBENCH_PYTHON:-uv run --no-sync python}"
CFG="${CURVE_CONFIG:-configs/curves/human_additive.yaml}"
CACHE=outputs/curves/pools
JOBS=outputs/curves/jobs.txt

case "${1:-}" in
plan)     $PY scripts/build_curve_plan.py --config "$CFG" --out "$JOBS" ;;
generate) $PY scripts/build_curve_plan.py --config "$CFG" --out "$JOBS" --stage generate >/dev/null
          n=$(wc -l < outputs/curves/generate.txt)
          $SUBMIT --name curvegen --cpu-only --cpus 8 --mem 96G --time 11:30:00 \
                  --array "1-${n}%10" -- bash -c '
            line=$(sed -n "${ALBENCH_TASK_ID}p" outputs/curves/generate.txt)
            [ -z "$line" ] && exit 0
            out=$(echo "$line" | sed -n "s/.*--out \([^ ]*\).*/\1/p")
            [ -s "$out" ] && { echo "SKIP $out"; exit 0; }
            eval "${ALBENCH_PYTHON:-uv run --no-sync python} -m albench.cli ${line#albench }"' ;;
label)    n=$(ls $CACHE/*.npz 2>/dev/null | grep -vc labeled)
          [ "${n:-0}" -eq 0 ] && { echo "nothing to label; run generate first" >&2; exit 1; }
          $SUBMIT --name curvelabel --gpus 1 --cpus 8 --mem 96G --time 11:30:00 \
                  --array "1-${n}" -- bash -c '
            f=$(ls outputs/curves/pools/*.npz | grep -v labeled | sed -n "${ALBENCH_TASK_ID}p")
            [ -z "$f" ] && exit 0
            ${ALBENCH_PYTHON:-uv run --no-sync python} scripts/label_pool.py \
              --in "$f" --out "${f%.npz}__labeled.npz"' ;;
link)     $PY scripts/link_screen_pools.py --cache outputs/curves/pools \
             --out outputs/curves/pools_linked --glob "*__labeled.npz" ;;
train)    $PY scripts/build_curve_plan.py --config "$CFG" --out "$JOBS" --stage train >/dev/null
          n=$(wc -l < "$JOBS")
          ALBENCH_GPU_CONSTRAINT= $SUBMIT --name curvetrain --gpus 1 --cpus 8 --mem 64G \
                  --time 11:30:00 --array "1-${n}%20" -- bash -c '
            line=$(sed -n "${ALBENCH_TASK_ID}p" outputs/curves/jobs.txt)
            [ -z "$line" ] && exit 0
            out=$(echo "$line" | sed -n "s/.*--output-dir \([^ ]*\).*/\1/p")
            if [ -n "$(find "$out" -name result.json -size +0c -print -quit 2>/dev/null)" ]; then
              echo "SKIP $out"; exit 0; fi
            eval "${ALBENCH_PYTHON:-uv run --no-sync python} $line"' ;;
status)   $PY scripts/build_curve_plan.py --config "$CFG" --out "$JOBS" --stage status ;;
*)        sed -n '2,10p' "$0"; exit 2 ;;
esac
