#!/bin/bash
#SBATCH --job-name=hp_watchdog
#SBATCH --partition=cpuq
#SBATCH --qos=cpu_snice
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH --output=outputs/analysis/watchdog_%j.log
#SBATCH --requeue

# Periodically re-runs submit_step1_bakeoff.py so cells that TIMEOUT on default's
# 12h walltime get re-queued. The submitter is idempotent — it skips cells that
# are already inflight or marked .bakeoff_done, and submits the rest.
#
# Runs every 30 min for 48h (cpuq default qos cap). Stop with `scancel $JOBID`.

cd /grid/wsbs/home_norepl/christen/ALBench-S2F
export TQDM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/cm/shared/apps/slurm/current/bin:$PATH

ITER=0
while true; do
  ITER=$((ITER + 1))
  echo "=== watchdog iter=${ITER}  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  # All 8 strategies × 3 seeds at D=300k genomic.
  STEP1_PREFER_DEFAULT=1 \
    STEP1_DS=300000 STEP1_RESERVOIRS=genomic STEP1_MODEL_BUDGET=50 \
    STEP1_SEEDS=42:0,43:1,44:2 \
    STEP1_STRATS=evo_single,evo_batch,evo_exploit,evo_adaptive,optuna_gp,optuna_tpe,llm_explore_nv1,llm_diverse_nv1 \
    uv run --no-sync python scripts/submit_step1_bakeoff.py 2>&1 \
      | grep -E "Submitted|Skipped|AlreadyDone|AllTiersFull|Queue counts|ERR"
  echo
  sleep 1800   # 30 min
done
