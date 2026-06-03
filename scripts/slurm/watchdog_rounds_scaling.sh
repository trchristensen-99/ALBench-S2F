#!/bin/bash
# Controller for the ROUNDS-SCALING study, H100-preferred with V100 backfill.
#
# The main 240-cell HP search is H100-only, so idle V100s are a no-contention pool.
# We use them for a head start on the cheap (D=30k) cells, then rotate everything to
# H100 the moment the main run frees the fast GPUs (see feedback-h100-preference).
#
#   PHASE A (now, concurrent with main search) — every 20 min, submit/resubmit the
#     D=30k ALGO cells on V100 (slow_nice). Only the algo variant runs here: the 3 LLM
#     variants share the Claude API rate limit with the running main search, so we defer
#     them to PHASE B to avoid contending on API quota. Resumes from *_meta.json
#     checkpoints. D=300k is intentionally NOT started on V100 (too GPU-bound — slower
#     than waiting). Continues until the main search drains (0 hp_* jobs, 2 consecutive
#     clean passes — which also means hp_watchdog exited and hp_autodeploy committed the
#     novel-axes change).
#
#   ROTATE — cancel any still-running V100 rsc_* jobs (their cells were incomplete;
#     they will resume on H100 from checkpoint, losing at most one in-flight model).
#
#   PHASE B (H100) — every 20 min, submit/resubmit ALL incomplete cells (D=30k resume +
#     D=300k fresh) on H100. Exits once 0 rsc_* jobs remain.
#
# Submit:  /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/watchdog_rounds_scaling.sh
# Cancel:  scancel --name=rsc_watchdog
#
#SBATCH --job-name=rsc_watchdog
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

INTERVAL=1200          # 20 min
MAX_PASSES=504         # 7 days / 20 min
NEED_CLEAN=2           # consecutive all-clear passes before declaring main run done

count_hp() { squeue --me -h -o %j 2>/dev/null | grep -c '^hp_'; }
count_rsc() { squeue --me -h -o %j 2>/dev/null | grep '^rsc_' | grep -v '^rsc_watchdog' | wc -l; }

echo "=== rsc_watchdog start $(date) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} ==="

# ---- PHASE A: V100 backfill of D=30k cells, until the main search finishes ----
echo "--- PHASE A: V100 backfill (D=30k, algo only) while main H100 search runs ---"
clean=0
for ((pass=1; pass<=MAX_PASSES; pass++)); do
  RSC_GPU=v100 RSC_DS=30000 RSC_VARIANTS=algo "$PY" "$REPO/scripts/submit_rounds_scaling.py" 2>&1 | tail -8 \
    || echo "  (V100 launcher pass errored — retry next interval)"
  hp=$(count_hp)
  echo "[phaseA] pass $pass $(date) :: hp_ jobs=$hp clean_streak=$clean"
  if [ "$hp" -eq 0 ]; then clean=$((clean + 1)); else clean=0; fi
  if [ "$clean" -ge "$NEED_CLEAN" ]; then
    echo "=== main search complete — rotating to H100 $(date) ==="
    break
  fi
  sleep "$INTERVAL"
done
if [ "$clean" -lt "$NEED_CLEAN" ]; then
  echo "!!! PHASE A hit MAX_PASSES without main-run completion — exiting, resubmit if needed."
  exit 1
fi

# ---- ROTATE: cancel remaining V100 rsc_ jobs so they re-run on H100 from checkpoint ----
echo "--- ROTATE: cancelling leftover V100 rsc_ jobs ---"
ids=$(squeue --me -h -o "%i %j %b" 2>/dev/null | grep ' rsc_' | grep -v 'rsc_watchdog' | grep 'v100' | awk '{print $1}')
if [ -n "$ids" ]; then echo "  cancelling: $ids"; scancel $ids; sleep 30; else echo "  (none running on V100)"; fi

# ---- PHASE B: H100 for all D (30k resume + 300k fresh) ----
echo "--- PHASE B: H100 for all cells ---"
for ((pass=1; pass<=MAX_PASSES; pass++)); do
  echo "--- phaseB pass $pass $(date) ---"
  RSC_GPU=h100 RSC_DS=30000,300000 "$PY" "$REPO/scripts/submit_rounds_scaling.py" 2>&1 | tail -10 \
    || echo "  (H100 launcher pass errored — retry next interval)"
  remaining=$(count_rsc)
  echo "  remaining rsc_ jobs in queue: $remaining"
  if [ "$remaining" -eq 0 ]; then
    RSC_GPU=h100 RSC_DS=30000,300000 "$PY" "$REPO/scripts/submit_rounds_scaling.py" 2>&1 | tail -4 || true
    remaining=$(count_rsc)
    if [ "$remaining" -eq 0 ]; then
      echo "=== ROUNDS-SCALING STUDY COMPLETE — rsc_watchdog exiting $(date) ==="
      exit 0
    fi
  fi
  sleep "$INTERVAL"
done
echo "=== rsc_watchdog reached MAX_PASSES; exiting $(date). Resubmit if incomplete. ==="
