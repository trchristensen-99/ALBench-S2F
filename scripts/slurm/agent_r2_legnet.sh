#!/bin/bash
# AutoResearch round 2: submit subagent-proposed LegNet configs to fast/default queues.
# Two cells: D=20k (fast queue, 4h limit) and D=100k (default queue, 10h limit).
#
# Each cell runs N configs in parallel via parallel_gpu_runner on a single GPU.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

submit_one() {
    local d=$1
    local proposals=$2
    local qos=$3
    local timelimit=$4
    local k_parallel=$5
    local jobname="agent_r2_leg_d${d}"
    local out_dir="results/preflight/hpsearch/agent_legnet_d${d}_r2"
    mkdir -p "$out_dir"

    source .venv/bin/activate
    python -m scripts.preflight.hpsearch._convert_agent_proposals \
        --proposals "$proposals" \
        --arch legnet \
        --d_train "$d" \
        --output_dir "$out_dir" \
        --epochs 60 \
        --patience 10

    local jobfile=$(mktemp)
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=$qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=$timelimit
#SBATCH --mem=140G

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache
export WANDB_PROJECT=albench-s2f-hpsearch
export WANDB_ENTITY=trchristensen99-cold-spring-harbor-laboratory

uv run --no-sync python scripts/preflight/parallel_gpu_runner.py "$out_dir/configs.json" "$k_parallel"
EOF
    local jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "  $jobname (qos=$qos) → $jid"
}

echo "Submitting AutoResearch round 2 jobs..."
submit_one 20000 results/preflight/hpsearch/agent_proposals_legnet_d20k_r2.json fast "03:30:00" 4
submit_one 100000 results/preflight/hpsearch/agent_proposals_legnet_d100k_r2.json default "10:00:00" 4
