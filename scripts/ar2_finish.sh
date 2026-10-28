#!/bin/bash
# Cron helper (every 10 min on the HPC login node): when an ar2 search has finished (ws/DONE or ws/STALLED) and its menu has not been exported yet, write the report and the
# menu (ws/report.md, ws/menu/menu_ar2.json); the menu_retrain cells of register_ar2menu.py wait for that file. Light: sqlite reads and a few file writes.
cd /grid/wsbs/home_norepl/christen/ALBench-S2F
source .venv/bin/activate; export PYTHONPATH=$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# keep-alive: an orchestrator whose tmux session died (crash) before its search finished is started again (it resumes from the database)
for cell in genomic_10k genomic_30k random_30k motif_30k; do
  w=outputs/ar2/ws_$cell
  if [ ! -f $w/DONE ] && [ ! -f $w/STALLED ] && ! tmux has-session -t ar2_$cell 2>/dev/null; then
    echo "$(date +%F\ %T) restarting orchestrator $cell"; ./scripts/ar2_launch.sh $cell 2>&1 | tail -1
  fi
done
for ws in outputs/ar2/ws_genomic_10k outputs/ar2/ws_genomic_30k outputs/ar2/ws_random_30k outputs/ar2/ws_motif_30k; do
  if { [ -f $ws/DONE ] || [ -f $ws/STALLED ]; } && [ ! -f $ws/menu/menu_ar2.json ]; then
    echo "$(date +%F\ %T) finishing $ws"
    python -m ar2 report --ws $ws --out $ws/report.md 2>&1 | tail -2
    python -m ar2 export-menu --ws $ws --out $ws/menu 2>&1 | tail -2
  fi
done
