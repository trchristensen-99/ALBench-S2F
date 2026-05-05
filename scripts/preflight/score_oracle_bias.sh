#!/bin/bash
# Submit the AG-S2 ensemble bias eval. Single-GPU job, ~1h wall:
# - 10 folds × ~5 panels × ~7 GC levels = 50-60 inference passes
# - Each panel ~500-2000 seqs at 42 batches/s = 10-50 sec
# - Total ~30-60 min including model loads.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cat > /tmp/_pf_bias_eval.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=pf_bias_eval
#SBATCH --output=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.out
#SBATCH --error=/grid/wsbs/home_norepl/christen/ALBench-S2F/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=128G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python scripts/preflight/score_oracle_bias.py
EOF
/cm/shared/apps/slurm/current/bin/sbatch /tmp/_pf_bias_eval.sh
rm -f /tmp/_pf_bias_eval.sh
