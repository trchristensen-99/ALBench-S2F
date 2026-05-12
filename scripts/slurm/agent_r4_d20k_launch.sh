#!/bin/bash
# Launch R4 D=20k AutoResearch around the new R2 winner
# (r2_explore_d100k_width512_d4: val=0.610 test=0.482, lr=3e-4 bs=128 drop=0.1).
# 28 configs split into exploit (12) / ablate-new-HPs (8) / explore (8).
# Uses new conv_dropout / dense_dropout / dense_dims HPs from commit b79e0e0.
#
# k_parallel=3 + EVAL_BATCH_MULT=2 for memory safety.
# Submits as dependency on R3 retry (2200534) so they don't fight for GPU.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

OUT_DIR=results/preflight/hpsearch/agent_legnet_d20000_r4
mkdir -p "$OUT_DIR"

source .venv/bin/activate
python -m scripts.preflight.hpsearch._convert_agent_proposals \
    --proposals results/preflight/hpsearch/agent_proposals_legnet_d20k_r4.json \
    --arch legnet \
    --d_train 20000 \
    --output_dir "$OUT_DIR" \
    --epochs 60 \
    --patience 10

DEP_FLAG=""
if [ -n "${DEP_JOB:-}" ]; then
    DEP_FLAG="--dependency=afterany:$DEP_JOB"
fi

JOBFILE=$(mktemp)
cat > "$JOBFILE" <<EOF
#!/bin/bash
#SBATCH --job-name=agent_r4_leg_d20k
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=10:00:00
#SBATCH --mem=140G

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache
export EVAL_BATCH_MULT=2
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

uv run --no-sync python -u scripts/preflight/parallel_gpu_runner.py "$OUT_DIR/configs.json" 3
EOF
if [ -n "$DEP_FLAG" ]; then
    JID=$($SBATCH --parsable $DEP_FLAG "$JOBFILE")
else
    JID=$($SBATCH --parsable "$JOBFILE")
fi
rm -f "$JOBFILE"
echo "agent_r4_leg_d20k → $JID${DEP_FLAG:+ ($DEP_FLAG)}"
