#!/bin/bash
# Run subagent-proposed LegNet D=20k configs (autoresearch round 1, programmatic).
# 30 configs in parallel_gpu_runner with k=4. Dependent on fill-in shootout
# finishing so we don't double-book slots.
#
# Usage: bash agent_proposals_legnet_d20k.sh [parent_jid]

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

PARENT_JID=${1:-2197371}  # fill-in shootout default
PROPOSALS=results/preflight/hpsearch/agent_proposals_legnet_d20k.json
OUT_DIR=results/preflight/hpsearch/agent_legnet_d20k_r1

# Convert proposals → parallel_gpu_runner configs.json
source .venv/bin/activate
python -m scripts.preflight.hpsearch._convert_agent_proposals \
    --proposals "$PROPOSALS" \
    --arch legnet \
    --d_train 20000 \
    --output_dir "$OUT_DIR" \
    --epochs 60 \
    --patience 10

jobfile=$(mktemp)
cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=agent_leg_d20k_r1
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=08:00:00
#SBATCH --mem=140G
#SBATCH --dependency=afterany:$PARENT_JID

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

uv run --no-sync python scripts/preflight/parallel_gpu_runner.py "$OUT_DIR/configs.json" 4
EOF

jid=$($SBATCH --parsable "$jobfile")
rm -f "$jobfile"
echo "  agent_leg_d20k_r1 (dep on $PARENT_JID) → $jid"
