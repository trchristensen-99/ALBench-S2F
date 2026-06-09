#!/bin/bash
# Fast gate: verify EvoAug determinism (#49) and that the canonical AG_S2 oracle
# loads + predicts on-GPU before launching the battery re-score chain. Routed to
# an (often idle) V100 so it starts immediately instead of queueing behind H100 work.
#
#   export PATH=/cm/shared/apps/slurm/current/bin:$PATH
#   sbatch scripts/slurm/oracle_smoke_v100.sh
#SBATCH --job-name=oracle_smoke
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
source scripts/slurm/setup_hpc_deps.sh
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export TQDM_DISABLE=1

echo "=== #49 EvoAug determinism ==="
uv run --no-sync python scripts/debug/test_evoaug_determinism.py
echo "=== AG_S2 oracle load+predict smoke ==="
uv run --no-sync python scripts/debug/test_ag_s2_oracle_predict.py
echo "=== ALL SMOKE TESTS PASSED ==="
