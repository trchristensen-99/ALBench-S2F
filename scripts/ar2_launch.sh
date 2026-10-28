#!/bin/bash
# Start one ar2 search: scripts/ar2_launch.sh <cell> [extra --set key=value ...]
# cell in genomic_10k genomic_30k genomic_100k random_30k motif_30k; D and the train manifest come from the cell name. The orchestrator runs in a tmux
# session ar2_<cell> on the login node (light: SQLite, a few CLI calls); training runs on GPU workers registered with register_ar2.py.
set -euo pipefail
cell="$1"; shift || true
R=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$R"
D=$(echo "$cell" | sed -E 's/.*_([0-9]+)k/\1/')
ws="outputs/ar2/ws_${cell}"
data=outputs/ar2/data
source .venv/bin/activate
set -a; . ~/.config/albench/llm.env; set +a
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TQDM_DISABLE=1
if [ ! -f "$ws/config.yaml" ]; then
  python -m ar2 init --ws "$ws" --profile deployment \
    --set data.D=$((D*1000)) --set data.train_manifest=$R/$data/train_${cell}.json \
    --set data.val_loop_manifest=$R/$data/val_loop.json --set data.val_select_manifest=$R/$data/val_select.json "$@"
fi
tmux has-session -t "ar2_${cell}" 2>/dev/null || tmux new -d -s "ar2_${cell}" "cd $R && source .venv/bin/activate && set -a && . ~/.config/albench/llm.env && set +a && export PYTHONPATH=$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TQDM_DISABLE=1 && python -m ar2 run --ws $ws >> $ws/run.log 2>&1"
echo "started ar2_${cell}"
