#!/bin/bash
# Launch R3 D=20k AutoResearch agent run.
# 27 configs around the R2 winner (lr=0.007, bs=1024, drop=0.05, wd=0.05, eff).
# Single GPU + k_parallel=4 — safe vs k=8 (mixed-size models can OOM at k=8).
#
# Run from REPO root on HPC after pulling latest. Submits to default queue.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

OUT_DIR=results/preflight/hpsearch/agent_legnet_d20000_r3
mkdir -p "$OUT_DIR"

source .venv/bin/activate
python -m scripts.preflight.hpsearch._convert_agent_proposals \
    --proposals results/preflight/hpsearch/agent_proposals_legnet_d20k_r3.json \
    --arch legnet \
    --d_train 20000 \
    --output_dir "$OUT_DIR" \
    --epochs 60 \
    --patience 10

JOBFILE=$(mktemp)
cat > "$JOBFILE" <<EOF
#!/bin/bash
#SBATCH --job-name=agent_r3_leg_d20k
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
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

uv run --no-sync python -u scripts/preflight/parallel_gpu_runner.py "$OUT_DIR/configs.json" 4
EOF
JID=$($SBATCH --parsable "$JOBFILE")
rm -f "$JOBFILE"
echo "agent_r3_leg_d20k → $JID"
