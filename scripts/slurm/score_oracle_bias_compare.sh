#!/bin/bash
# Score the chr-split test/control battery with ONE explicit AG_S2 oracle ensemble,
# for the designed-sequence bias comparison. Driven by the autodeploy_oracle_compare
# controller, which submits this twice (full856k_clean and no_designed).
#
# Env (set via --export):
#   CMP_ORACLE_DIR  — abs path to dir with fold_*/best_model/checkpoint
#   CMP_OUT_DIR     — abs path for *_oracle.npz (per-oracle, never the live test dir)
#
# Idempotent: exits 0 immediately if all 4 npz already present (resume-safe).
#
#SBATCH --job-name=oraclecmp
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --exclude=bamgpu15,bamgpu25
#SBATCH --export=ALL
set -uo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
: "${CMP_ORACLE_DIR:?set CMP_ORACLE_DIR}"
: "${CMP_OUT_DIR:?set CMP_OUT_DIR}"

set +u; source /etc/profile.d/modules.sh; set -u
module load EB5 2>/dev/null || true
cd "$REPO" || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export PYTHONUNBUFFERED=1

echo "=== oraclecmp node=${SLURMD_NODENAME:-?} oracle=$CMP_ORACLE_DIR out=$CMP_OUT_DIR $(date) ==="

# Safety: the oracle ensemble must have all 10 folds.
N_FOLDS=$(find "$CMP_ORACLE_DIR" -maxdepth 3 -type d -path "*/best_model/checkpoint" 2>/dev/null | wc -l)
if [ "$N_FOLDS" -lt 10 ]; then
  echo "ERROR: $CMP_ORACLE_DIR has only ${N_FOLDS}/10 folds — aborting"; exit 2
fi

# Idempotent skip.
need=0
for fn in genomic_oracle.npz snv_oracle.npz ood_oracle.npz random_10k_oracle.npz; do
  [ -f "$CMP_OUT_DIR/$fn" ] || need=1
done
if [ "$need" -eq 0 ]; then
  echo "=== all 4 npz already present in $CMP_OUT_DIR — nothing to do ==="; exit 0
fi

uv run --no-sync python scripts/generate_ag_s2_test_labels.py \
  --oracle-dir "$CMP_ORACLE_DIR" \
  --out-dir "$CMP_OUT_DIR"
rc=$?
if [ $rc -eq 0 ]; then echo "=== DONE rc=0 $(date) ==="; else echo "=== FAILED rc=$rc ==="; fi
exit $rc
