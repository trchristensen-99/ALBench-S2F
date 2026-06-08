#!/bin/bash
# Self-healing watchdog for the Step-1 deep bake-off, D=300k tier (s1_*_d300000 jobs).
#
# Identical to watchdog_step1_bakeoff.sh but exports STEP1_DS=300000 so the idempotent
# launcher (scripts/submit_step1_bakeoff.py) targets the 300k cells. Cells are namespaced
# k562_<res>_d300000/seed<ds>_<hs>/<variant> -- fully independent of the completed d30000
# tree -- so this never touches finished d30000 compute. Each cell resumes from per-model
# *_meta.json checkpoints; the launcher skips in-flight + >=100-meta cells.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/watchdog_step1_bakeoff_d300k.sh
# Cancel:
#   scancel --name=s1d3_watchdog
#
#SBATCH --job-name=s1d3_watchdog
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --partition=cpuq
#SBATCH --qos=cpu_snice
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=7-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -uo pipefail
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
PY="$REPO/.venv/bin/python"
cd "$REPO" || exit 1

export STEP1_DS=300000     # D=300k tier
# LEAN D=300k confirmation set: recipe is already frozen at D=30k, so we only
# re-run the strategies needed to CONFIRM it + the random baseline. optuna_tpe /
# ray_asha / ray_bohb are dropped (0-1 knee votes at D=30k, expensive) so the
# launcher never resubmits them for dinuc/motif. Genomic's optuna/ray already
# completed (own .bakeoff_done) and are unaffected. ROUNDS/PER_ROUND unchanged
# (50x2) so evo search dynamics stay identical to D=30k -> no search bias.
export STEP1_STRATS=random,evo_single,evo_batch,evo_explore,evo_exploit,evo_massive,evo_adaptive,evo_knowledgeable,llm_default,llm_diverse,llm_exploit

INTERVAL=1200          # 20 min between launcher passes
MAX_PASSES=504         # 7 days / 20 min — hard stop matching walltime

echo "=== s1d3_watchdog start $(date) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} STEP1_DS=$STEP1_DS ==="
for ((pass=1; pass<=MAX_PASSES; pass++)); do
  echo "--- pass $pass $(date) ---"
  "$PY" "$REPO/scripts/submit_step1_bakeoff.py" 2>&1 | tail -8 || echo "  (launcher pass errored — will retry next interval)"
  remaining=$(squeue --me -h -o %j 2>/dev/null | grep -c '_d300000')
  echo "  remaining d300000 bake-off jobs in queue: $remaining"
  if [ "$remaining" -le 0 ]; then
    # One confirming pass: if the launcher still queues nothing, all cells done.
    "$PY" "$REPO/scripts/submit_step1_bakeoff.py" 2>&1 | tail -3 || true
    remaining=$(squeue --me -h -o %j 2>/dev/null | grep -c '_d300000')
    if [ "$remaining" -le 0 ]; then
      echo "=== ALL D=300k BAKE-OFF CELLS COMPLETE — watchdog exiting $(date) ==="
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
echo "=== s1d3_watchdog reached MAX_PASSES; exiting $(date). Resubmit if bake-off incomplete. ==="
