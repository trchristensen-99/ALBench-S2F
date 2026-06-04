#!/bin/bash
# Auto-deploy the staged novel-axes feature once the mixed6 HP search finishes.
#
# Waits (on the cheap CPU tier) until 0 hp_ jobs remain in the queue — i.e. every
# HP-search cell is complete and the watchdog has exited — then drops the 3 staged
# files from novelaxes_staging/ into the live repo, validates them (py_compile +
# import + a functional wiring check), and makes a LOCAL git commit (NO push).
#
# Deployment is held until the run completes so we never touch files that 200+
# running jobs import mid-flight. A sentinel guards against double-deploy across
# requeues. On any validation failure the originals are restored and no commit is
# made.
#
# Submit:
#   /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/autodeploy_novel_axes.sh
# Cancel:
#   scancel --name=hp_autodeploy
#
#SBATCH --job-name=hp_autodeploy
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
STAGING=/grid/wsbs/home_norepl/christen/novelaxes_staging
PY="$REPO/.venv/bin/python"
SENTINEL="$STAGING/.deployed"
INTERVAL=1200          # 20 min between checks (matches watchdog cadence)
MAX_PASSES=504         # 7 days / 20 min — hard stop matching walltime
NEED_CLEAN=2           # consecutive all-clear passes required before deploying

FILES=(
  "experiments/scaling_hp_search.py"
  "experiments/llm_autoresearch.py"
  "models/legnet_student.py"
)

cd "$REPO" || exit 1
echo "=== hp_autodeploy start $(date) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} ==="

# Guard: already deployed (or permanently failed) -> nothing to do.
if [ -f "$SENTINEL" ]; then
  echo "Sentinel present ($SENTINEL) — already handled. Exiting."
  cat "$SENTINEL"
  exit 0
fi

# --- count hp_ jobs excluding the watchdog and this deploy job ---
count_hp_jobs() {
  squeue --me -h -o %j 2>/dev/null \
    | grep '^hp_' | grep -v '^hp_watchdog' | grep -v '^hp_autodeploy' \
    | wc -l
}

clean_passes=0
for ((pass=1; pass<=MAX_PASSES; pass++)); do
  remaining=$(count_hp_jobs)
  wd=$(squeue --me -h -o %j 2>/dev/null | grep -c '^hp_watchdog')
  echo "--- pass $pass $(date) :: hp_ jobs=$remaining watchdog=$wd clean_streak=$clean_passes ---"
  if [ "$remaining" -eq 0 ]; then
    clean_passes=$((clean_passes + 1))
  else
    clean_passes=0
  fi
  if [ "$clean_passes" -ge "$NEED_CLEAN" ]; then
    echo "=== HP search complete ($NEED_CLEAN consecutive all-clear passes) — deploying $(date) ==="
    break
  fi
  sleep "$INTERVAL"
done

if [ "$clean_passes" -lt "$NEED_CLEAN" ]; then
  echo "!!! reached MAX_PASSES without confirming completion — NOT deploying. Resubmit if needed."
  exit 1
fi

# --- pre-flight: staged files must all exist ---
for f in "${FILES[@]}"; do
  if [ ! -f "$STAGING/$f" ]; then
    echo "!!! staged file missing: $STAGING/$f — aborting deploy."
    exit 1
  fi
done

# --- backup live originals, then drop in staged copies ---
TS=$(date +%Y%m%d_%H%M%S)
BACKUP="$STAGING/live_backup_$TS"
mkdir -p "$BACKUP"
echo "Backing up live originals to $BACKUP"
for f in "${FILES[@]}"; do
  mkdir -p "$BACKUP/$(dirname "$f")"
  cp -p "$REPO/$f" "$BACKUP/$f"
  cp -p "$STAGING/$f" "$REPO/$f"
  echo "  deployed $f"
done

restore() {
  echo "!!! restoring live originals from backup"
  for f in "${FILES[@]}"; do cp -p "$BACKUP/$f" "$REPO/$f"; done
}

# --- validate: py_compile + import + functional wiring check ---
echo "=== validating $(date) ==="
ok=1
for f in "${FILES[@]}"; do
  "$PY" -m py_compile "$REPO/$f" || { echo "  py_compile FAILED: $f"; ok=0; }
done

if [ "$ok" -eq 1 ]; then
  "$PY" - <<'PYEOF'
import sys
from experiments.scaling_hp_search import apply_experimental_knobs, EXPERIMENTAL_KNOBS
from experiments.llm_autoresearch import dict_to_hpconfig
hp = dict_to_hpconfig({"lr": 1e-3, "activation": "gelu", "made_up": 1}, seed=1, allow_novel=True)
assert hp.extra.get("activation") == "gelu", "novel key not gathered"
assert hp.extra.get("made_up") == 1, "unknown key not recorded"
hp_off = dict_to_hpconfig({"lr": 1e-3, "activation": "gelu"}, seed=1, allow_novel=False)
assert hp_off.extra == {}, "OFF path leaked extra"
tr, md, ls, applied, recorded = apply_experimental_knobs(
    {"loss": "huber", "huber_delta": 99, "activation": "banana", "se_reduction": 999}
)
assert ls["loss"] == "huber" and ls["huber_delta"] == 5.0, "loss knob misrouted/unclipped"
assert md["activation"] == "silu", "bad activation not defaulted"
assert md["se_reduction"] == 16, "se_reduction not clipped"
print("FUNCTIONAL OK")
PYEOF
  [ $? -eq 0 ] || { echo "  functional check FAILED"; ok=0; }
fi

if [ "$ok" -ne 1 ]; then
  restore
  echo "DEPLOY FAILED (validation) $(date)" > "$SENTINEL"
  echo "!!! validation failed — originals restored, NO commit. Sentinel marks FAILED to avoid retry loop."
  exit 1
fi

# --- local commit (NO push) ---
echo "=== committing locally (no push) $(date) ==="
cd "$REPO" || { restore; exit 1; }
git add "${FILES[@]}"
git commit -m "$(cat <<'MSG'
Deploy novel-axes HP feature (LLM_ALLOW_NOVEL_AXES, default OFF)

Auto-deployed by scripts/slurm/autodeploy_novel_axes.sh after the mixed6
HP search completed. Behavior is byte-identical to the 15-axis search
unless LLM_ALLOW_NOVEL_AXES=1 is set in the LLM-strategy job env.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
MSG
)" && echo "commit OK" || { echo "!!! git commit failed"; restore; echo "DEPLOY FAILED (git) $(date)" > "$SENTINEL"; exit 1; }

COMMIT=$(git rev-parse HEAD)
{
  echo "DEPLOYED $(date)"
  echo "commit=$COMMIT"
  echo "backup=$BACKUP"
} > "$SENTINEL"
echo "=== novel-axes feature DEPLOYED locally (commit $COMMIT). NOT pushed — push manually. $(date) ==="
exit 0
