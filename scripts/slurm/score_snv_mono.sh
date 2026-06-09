#!/bin/bash
# Score the strict-monoallelic chr 7+13 SNV pairs (~29.4k) with the canonical AG_S2
# oracle (outputs/oracle_full856k_clean/s2), replacing the over-sized snv_oracle.npz.
# The scorer first verifies the oracle reproduces the existing genomic battery, then
# writes snv_oracle.npz stamped test_set_version=snv_mono_chrsplit_v1.
#
# Launch from the login node:
#   export PATH=/cm/shared/apps/slurm/current/bin:$PATH
#   sbatch scripts/slurm/score_snv_mono.sh
#SBATCH --job-name=score_snv_mono
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export TQDM_DISABLE=1
export PYTHONUNBUFFERED=1

echo "=== Build mono SNV intermediate (idempotent) — $(date) ==="
uv run --no-sync python scripts/build_chrsplit_snv_mono.py

echo "=== Score mono SNV with canonical AG_S2 oracle — $(date) ==="
uv run --no-sync python scripts/score_chrsplit_snv_mono_ag_s2.py

echo "=== Done rc=$? — $(date) ==="
