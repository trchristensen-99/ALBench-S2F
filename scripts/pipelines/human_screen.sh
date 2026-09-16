#!/bin/bash
# The human 30k parameter screen, end to end, on whatever scheduler you have.
#
# Each stage submits through scripts/cluster/submit.sh, so the only site-specific
# knowledge lives in scripts/cluster/site.env. With no scheduler present the stages
# run in the foreground instead, which is slow but correct.
#
#   ./scripts/pipelines/human_screen.sh enumerate   # write the cell list
#   ./scripts/pipelines/human_screen.sh generate    # sequences for every cell
#   ./scripts/pipelines/human_screen.sh label       # oracle labels (needs the oracle)
#   ./scripts/pipelines/human_screen.sh link        # expose cells as pools
#   ./scripts/pipelines/human_screen.sh train       # one student per cell
#   ./scripts/pipelines/human_screen.sh status
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
SUBMIT=scripts/cluster/submit.sh
PY="${ALBENCH_PYTHON:-uv run --no-sync python}"
CONFIG="${SCREEN_CONFIG:-configs/screen/human_30k_300k.yaml}"
CACHE=outputs/screen/cache
CELLS=outputs/screen/train_cells.txt

case "${1:-}" in
enumerate)
  $PY -m albench.cli screen --config "$CONFIG" --write outputs/screen/jobs.sh
  echo "wrote outputs/screen/jobs.sh ($(wc -l < outputs/screen/jobs.sh) cells)"
  ;;

generate)
  n=$(wc -l < outputs/screen/jobs.sh)
  # Idempotent per task: a cell that already has its npz is skipped, so the array can
  # be resubmitted freely after preemption.
  $SUBMIT --name screengen --cpu-only --cpus 4 --mem 48G --time 08:00:00 \
          --array "1-${n}%36" -- bash -c '
    line=$(sed -n "${ALBENCH_TASK_ID}p" outputs/screen/jobs.sh)
    [ -z "$line" ] && exit 0
    out=$(echo "$line" | sed -n "s/.*--out \([^ ]*\).*/\1/p")
    [ -s "$out" ] && { echo "SKIP $out"; exit 0; }
    eval "${ALBENCH_PYTHON:-uv run --no-sync python} -m albench.cli ${line#albench }"'
  ;;

label)
  n=$(ls $CACHE/*__d30000__*.npz 2>/dev/null | grep -vc labeled)
  [ "$n" -eq 0 ] && { echo "no unlabelled D=30k cells found; run generate first" >&2; exit 1; }
  $SUBMIT --name screenlabel --gpus 1 --cpus 8 --mem 96G --time 11:30:00 \
          --array "1-${n}" -- bash -c '
    f=$(ls outputs/screen/cache/*__d30000__*.npz | grep -v labeled | sed -n "${ALBENCH_TASK_ID}p")
    [ -z "$f" ] && exit 0
    out="${f%.npz}__labeled.npz"
    [ -s "$out" ] && { echo "SKIP $out"; exit 0; }
    ${ALBENCH_PYTHON:-uv run --no-sync python} scripts/label_pool.py --in "$f" --out "$out"'
  ;;

link)
  $PY scripts/link_screen_pools.py
  ls outputs/screen/pools | grep -v manifest.json | sort > "$CELLS"
  echo "linked $(wc -l < "$CELLS") cells"
  ;;

train)
  [ -s "$CELLS" ] || { echo "no $CELLS; run link first" >&2; exit 1; }
  n=$(wc -l < "$CELLS")
  # No GPU constraint: the student is a CNN and pays only ~2.4x on an older card, so
  # it is better off taking whatever is idle than queueing for a specific model.
  ALBENCH_GPU_CONSTRAINT= $SUBMIT --name sctrain --gpus 1 --cpus 8 --mem 64G \
          --time 11:30:00 --array "1-${n}%20" -- bash -c '
    cell=$(sed -n "${ALBENCH_TASK_ID}p" outputs/screen/train_cells.txt)
    [ -z "$cell" ] && exit 0
    out=outputs/screen/train/$cell
    # The driver writes <cell>/n<N>/hp<i>/seed<s>/result.json, NOT <cell>/results.json.
    if [ -n "$(find "$out" -name result.json -size +0c -print -quit 2>/dev/null)" ]; then
      echo "SKIP already done: $cell"; exit 0
    fi
    ${ALBENCH_PYTHON:-uv run --no-sync python} experiments/exp1_1_scaling.py \
      --task k562 --student legnet --oracle ag_s2 --reservoir "$cell" \
      --pool-base-dir outputs/screen/pools --training-sizes 30000 \
      --n-replicates 1 --no-hp-sweep --chr-split --save-predictions --output-dir "$out"'
  ;;

status)
  # Counting has to survive a fresh clone where none of these paths exist yet, which
  # is the state a new collaborator is actually in.
  count_lines () { [ -s "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
  count_glob ()  { local n=0 f; for f in $1; do [ -e "$f" ] || continue; n=$((n+1)); done; echo "$n"; }
  gen=$(count_glob "$CACHE/*.npz"); lab=$(count_glob "$CACHE/*__labeled.npz")
  printf "  cells enumerated : %s\n" "$(count_lines outputs/screen/jobs.sh)"
  printf "  generated        : %s\n" "$((gen - lab))"
  printf "  labelled         : %s\n" "$lab"
  printf "  linked as pools  : %s\n" "$(count_lines "$CELLS")"
  d=0
  if [ -s "$CELLS" ]; then
    while read -r c; do
      [ -n "$(find "outputs/screen/train/$c" -name result.json -size +0c -print -quit 2>/dev/null)" ] && d=$((d+1))
    done < "$CELLS"
  fi
  printf "  students trained : %s\n" "$d"
  ;;

*) sed -n '2,14p' "$0"; exit 2;;
esac
