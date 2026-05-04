#!/bin/bash
# End-to-end pipeline: build ref+alt+boda2 K562 pool → run S2 fold inference
# (10 array tasks) → aggregate pseudolabels → fire Task 3 D_max sweep.
#
# Each step is its own SLURM job with --dependency=afterok on the previous.
# Submit once and walk away:
#   bash scripts/preflight/pipeline_cache_regen_then_task3.sh

set -euo pipefail

mkdir -p logs results/preflight

# ── Step 1: pool builder (~5 min CPU only) ────────────────────────────────
POOL_BUILD_SCRIPT=$(mktemp)
cat > "$POOL_BUILD_SCRIPT" <<'EOF'
#!/bin/bash
#SBATCH --job-name=pf_pool_build
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="$PWD"
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python scripts/preflight/build_k562_refalt_pool.py
EOF
POOL_JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable "$POOL_BUILD_SCRIPT")
echo "  step 1 (pool build) submitted: ${POOL_JOB}"
rm -f "$POOL_BUILD_SCRIPT"

# ── Step 2: per-fold inference (10 array tasks, ~6-10h each) ─────────────
FOLD_INFER_SCRIPT=$(mktemp)
cat > "$FOLD_INFER_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=pf_s2_fold_infer
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --array=0-9
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="\$PWD"
export TORCHDYNAMO_DISABLE=1
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python scripts/preflight/infer_s2_fold.py --fold \$SLURM_ARRAY_TASK_ID
EOF
FOLD_JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable \
    --dependency=afterok:${POOL_JOB} "$FOLD_INFER_SCRIPT")
echo "  step 2 (10-fold S2 inference array) submitted: ${FOLD_JOB} (deps on ${POOL_JOB})"
rm -f "$FOLD_INFER_SCRIPT"

# ── Step 3: aggregate (CPU, ~5 min, depends on all 10 fold tasks) ────────
AGG_SCRIPT=$(mktemp)
cat > "$AGG_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=pf_s2_aggregate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=64G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="\$PWD"
source scripts/slurm/setup_hpc_deps.sh
uv run --no-sync python scripts/preflight/aggregate_s2_pseudolabels.py
EOF
AGG_JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable \
    --dependency=afterok:${FOLD_JOB} "$AGG_SCRIPT")
echo "  step 3 (aggregate) submitted: ${AGG_JOB} (deps on all ${FOLD_JOB} array tasks)"
rm -f "$AGG_SCRIPT"

# ── Step 4: Task 3 D_max launcher (depends on aggregate) ────────────────
TASK3_SCRIPT=$(mktemp)
cat > "$TASK3_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=pf_task3_launcher
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --mem=8G
set -euo pipefail
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
bash scripts/preflight/task3_lr_bs_dmax.sh
EOF
TASK3_JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable \
    --dependency=afterok:${AGG_JOB} "$TASK3_SCRIPT")
echo "  step 4 (Task 3 D_max launcher) submitted: ${TASK3_JOB} (deps on ${AGG_JOB})"
rm -f "$TASK3_SCRIPT"

echo
echo "=== Pipeline submitted ==="
echo "  pool build  : ${POOL_JOB}"
echo "  fold infer  : ${FOLD_JOB} (array 0-9)"
echo "  aggregate   : ${AGG_JOB}"
echo "  task3 fire  : ${TASK3_JOB}"
echo
echo "Monitor:"
echo "  squeue -u \$USER -j ${POOL_JOB},${FOLD_JOB},${AGG_JOB},${TASK3_JOB}"
