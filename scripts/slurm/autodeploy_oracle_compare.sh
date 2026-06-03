#!/bin/bash
# Post-search auto-regen of the designed-sequence oracle-bias comparison.
#
# Held until the mixed6 HP search finishes so the GPU scoring never competes with
# the running search. Once 0 hp_ jobs remain (2 consecutive clean passes, like
# hp_autodeploy), it:
#   1. submits two short H100 scoring jobs (qos=fast) — the chr-split test/control
#      battery scored under each oracle into outputs/oracle_bias_compare/<variant>/:
#        • full856k_clean  (designed-INCLUDED, canonical)   outputs/oracle_full856k_clean/s2
#        • no_designed     (designed high-activity removed)  outputs/oracle_no_designed/s2
#   2. waits for both to finish (idempotent — skips a variant already scored),
#   3. runs scripts/analysis/plot_oracle_designed_bias.py to regenerate the
#      comparison figure + summary CSV.
#
# A sentinel guards against re-run across requeues. Scoring jobs are resume-safe,
# so a controller restart just re-checks and resubmits whatever is missing.
#
# Submit:  /cm/shared/apps/slurm/current/bin/sbatch scripts/slurm/autodeploy_oracle_compare.sh
# Cancel:  scancel --name=oraclecmp_ctl
#
#SBATCH --job-name=oraclecmp_ctl
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
SCORE_SH="$REPO/scripts/slurm/score_oracle_bias_compare.sh"
CMP_ROOT="$REPO/outputs/oracle_bias_compare"
SENTINEL="$CMP_ROOT/.done"
INTERVAL=1200          # 20 min
MAX_PASSES=504         # 7 days / 20 min
NEED_CLEAN=2

# variant : oracle-ensemble dir (each has fold_*/best_model/checkpoint)
declare -A ORACLE
ORACLE[full856k_clean]="$REPO/outputs/oracle_full856k_clean/s2"
ORACLE[no_designed]="$REPO/outputs/oracle_no_designed/s2"

cd "$REPO" || exit 1
echo "=== oraclecmp_ctl start $(date) node=${SLURMD_NODENAME:-?} job=${SLURM_JOB_ID:-?} ==="

if [ -f "$SENTINEL" ]; then
  echo "Sentinel present ($SENTINEL) — already handled. Exiting."; cat "$SENTINEL"; exit 0
fi

count_hp() { squeue --me -h -o %j 2>/dev/null | grep '^hp_' | grep -v '^hp_watchdog' | grep -v '^hp_autodeploy' | wc -l; }
count_score() { squeue --me -h -o %j 2>/dev/null | grep '^oraclecmp_' | grep -v '^oraclecmp_ctl' | wc -l; }
variant_done() {  # $1=variant ; all 4 npz present?
  local d="$CMP_ROOT/$1" fn
  for fn in genomic_oracle.npz snv_oracle.npz ood_oracle.npz random_10k_oracle.npz; do
    [ -f "$d/$fn" ] || return 1
  done
  return 0
}

# ---- wait for the main HP search to drain ----
clean=0
for ((pass=1; pass<=MAX_PASSES; pass++)); do
  hp=$(count_hp)
  echo "--- pass $pass $(date) :: hp_ jobs=$hp clean_streak=$clean ---"
  if [ "$hp" -eq 0 ]; then clean=$((clean + 1)); else clean=0; fi
  if [ "$clean" -ge "$NEED_CLEAN" ]; then
    echo "=== HP search complete — starting oracle-bias scoring $(date) ==="; break
  fi
  sleep "$INTERVAL"
done
if [ "$clean" -lt "$NEED_CLEAN" ]; then
  echo "!!! reached MAX_PASSES without completion — NOT scoring. Resubmit if needed."; exit 1
fi

# pre-flight: both oracle ensembles must exist with 10 folds
for v in full856k_clean no_designed; do
  nf=$(find "${ORACLE[$v]}" -maxdepth 3 -type d -path "*/best_model/checkpoint" 2>/dev/null | wc -l)
  if [ "$nf" -lt 10 ]; then
    echo "!!! ${ORACLE[$v]} has only ${nf}/10 folds — aborting."; exit 1
  fi
done

mkdir -p "$CMP_ROOT"

# ---- submit/resubmit scoring until both variants are scored ----
for ((pass=1; pass<=MAX_PASSES; pass++)); do
  pending=0
  for v in full856k_clean no_designed; do
    if variant_done "$v"; then
      echo "  [$v] already scored"; continue
    fi
    pending=1
    # only (re)submit if not already queued/running
    running=$(squeue --me -h -o %j 2>/dev/null | grep -c "^oraclecmp_${v}$")
    if [ "$running" -ge 1 ]; then echo "  [$v] in-flight"; continue; fi
    out="$CMP_ROOT/$v"; mkdir -p "$out"
    jid=$(sbatch --parsable \
      --job-name="oraclecmp_${v}" \
      --export=ALL,CMP_ORACLE_DIR="${ORACLE[$v]}",CMP_OUT_DIR="$out" \
      "$SCORE_SH" 2>&1)
    if [[ "$jid" =~ ^[0-9]+$ ]]; then echo "  [$v] submitted -> $jid"; else echo "  [$v] submit err: $jid"; fi
  done
  if [ "$pending" -eq 0 ]; then
    echo "=== both variants scored — generating comparison $(date) ==="; break
  fi
  echo "  scoring jobs in queue: $(count_score)"
  sleep "$INTERVAL"
done

if ! { variant_done full856k_clean && variant_done no_designed; }; then
  echo "!!! scoring did not complete in time — NOT plotting. Resubmit controller."; exit 1
fi

# ---- regenerate the comparison figure + CSV (CPU, in this job) ----
echo "=== plotting designed-bias comparison $(date) ==="
"$PY" "$REPO/scripts/analysis/plot_oracle_designed_bias.py" \
  --canonical-dir "$CMP_ROOT/full856k_clean" \
  --compare-dir   "$CMP_ROOT/no_designed" \
  --out-dir       "$REPO/results/diagnostics/oracle_designed_bias"
rc=$?
if [ $rc -ne 0 ]; then echo "!!! plot failed rc=$rc — NOT marking done."; exit 1; fi

{
  echo "DONE $(date)"
  echo "canonical=$CMP_ROOT/full856k_clean (oracle_full856k_clean/s2, designed-incl)"
  echo "compare=$CMP_ROOT/no_designed (oracle_no_designed/s2)"
  echo "figure=results/diagnostics/oracle_designed_bias/oracle_designed_bias.png"
} > "$SENTINEL"
echo "=== ORACLE-BIAS COMPARISON COMPLETE — oraclecmp_ctl exiting $(date) ==="
exit 0
