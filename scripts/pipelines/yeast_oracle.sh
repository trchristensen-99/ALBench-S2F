#!/bin/bash
# The yeast DREAM-RNN oracle: two variants, ten folds each, then the per-class report.
#
# Variant A trains on the bulk random library only. Variant B additionally trains on
# 80% of each splittable DREAM eval class, to measure what including a class buys on
# that class. Both are needed: the A-vs-B comparison is only meaningful on sequences
# BOTH held out, otherwise B is flattered by exactly what it memorised.
#
#   ./scripts/pipelines/yeast_oracle.sh prepare     # folds + eval-class map
#   ./scripts/pipelines/yeast_oracle.sh train A     # or: train B
#   ./scripts/pipelines/yeast_oracle.sh predict     # score eval classes, all 20 folds
#   ./scripts/pipelines/yeast_oracle.sh compare
#   ./scripts/pipelines/yeast_oracle.sh status
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
SUBMIT=scripts/cluster/submit.sh
PY="${ALBENCH_PYTHON:-uv run --no-sync python}"
OUT=outputs/yeast_oracle_v2
# Classes variant B may train on. The motif_* sets are excluded because the split
# builder marks them unsplittable: too few ref/alt components to hold a fold out
# without splitting a pair, which would destroy the allelic difference they measure.
B_CLASSES=random,challenging,high_expression,low_expression,native_genomic,snv

case "${1:-}" in
prepare)
  $PY scripts/build_yeast_folds.py && $PY scripts/build_yeast_eval_classes.py
  ;;

train)
  variant="${2:-A}"
  case "$variant" in
    A) tag=""; extra="";;
    B) tag="_B"; extra="--include-eval-classes $B_CLASSES";;
    *) echo "train takes A or B" >&2; exit 2;;
  esac
  # Recurrent training: constrain the accelerator. Measured ~7.6x slower on an older
  # card here, far worse than the ~2.4x a CNN pays.
  $SUBMIT --name "yoracle$tag" --gpus 1 --cpus 8 --mem 200G --time 11:30:00 \
          --array "0-9" -- bash -c "
    f=\$ALBENCH_TASK_ID
    [ -s \"$OUT/fold_\${f}$tag/test_metrics.json\" ] && { echo \"SKIP fold \$f$tag\"; exit 0; }
    ${ALBENCH_PYTHON:-uv run --no-sync python} experiments/train_yeast_oracle_v2.py \
      --fold-id \"\$f\" --tag '$tag' $extra --resume"
  ;;

predict)
  $SUBMIT --name yopred --gpus 1 --cpus 8 --mem 96G --time 04:00:00 \
          --array "0-19" -- bash -c "
    i=\$ALBENCH_TASK_ID
    if [ \"\$i\" -lt 10 ]; then f=\$i; tag=''; else f=\$((i-10)); tag='_B'; fi
    out=\"$OUT/fold_\${f}\${tag}/eval_class_predictions.npz\"
    [ -s \"\$out\" ] && { echo \"SKIP \$out\"; exit 0; }
    ${ALBENCH_PYTHON:-uv run --no-sync python} scripts/predict_yeast_eval_classes.py \
      --fold-id \"\$f\" --tag \"\$tag\""
  ;;

compare)
  $PY scripts/analysis/eval_yeast_oracle_by_class.py
  ;;

status)
  for v in "" _B; do
    n=0; for f in 0 1 2 3 4 5 6 7 8 9; do
      [ -s "$OUT/fold_${f}${v}/test_metrics.json" ] && n=$((n+1)); done
    p=0; for f in 0 1 2 3 4 5 6 7 8 9; do
      [ -s "$OUT/fold_${f}${v}/eval_class_predictions.npz" ] && p=$((p+1)); done
    printf "  variant %-2s trained %2d/10   scored %2d/10\n" "$(echo "${v:-A}" | tr -d _)" "$n" "$p"
  done
  ;;

*) sed -n '2,13p' "$0"; exit 2;;
esac
