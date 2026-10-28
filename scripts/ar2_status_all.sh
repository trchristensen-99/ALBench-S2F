#!/bin/bash
# Compact status of every ar2 search workspace (run on the login node; read-only, light).
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
source .venv/bin/activate; export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for ws in outputs/ar2/ws_*; do
  [ -f $ws/config.yaml ] || continue
  echo "=== $ws $(tmux has-session -t ar2_${ws#outputs/ar2/ws_} 2>/dev/null && echo '[orchestrator up]' || echo '[orchestrator DOWN]') $([ -f $ws/DONE ] && echo DONE) $([ -f $ws/STALLED ] && echo STALLED)"
  python -m ar2 status --ws $ws 2>&1 | grep -v "^$" | sed -n '2,12p' | cut -c1-190
  tail -n 1 $ws/run.log | cut -c1-200
done
