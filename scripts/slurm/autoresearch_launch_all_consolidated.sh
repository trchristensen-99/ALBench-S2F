#!/bin/bash
# Launch consolidated AutoResearch round-0 for all cells (skip DREAM-RNN).
# One SLURM job per (arch, D) cell. Each runs 15 configs (3 roles × 5) via
# parallel_gpu_runner on one GPU.
#
# Usage: bash scripts/slurm/autoresearch_launch_all_consolidated.sh [round_idx]

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch
ROUND_IDX=${1:-0}

declare -a CELLS=(
    "legnet 5000"
    "legnet 100000"
    "dream_attn 5000"
    "dream_attn 100000"
)

submit_one() {
    local arch=$1
    local d=$2
    local qos=$3
    local timelimit=$4
    local k_parallel=$5

    local jobname="ar_${arch:0:3}_d${d}_r${ROUND_IDX}"
    local jobfile=$(mktemp)
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=$qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=$timelimit
#SBATCH --mem=120G

export ARCH=$arch
export D_TRAIN=$d
export ROUND_IDX=$ROUND_IDX
export K_PARALLEL=$k_parallel

bash $REPO/scripts/slurm/autoresearch_cell_consolidated.sh
EOF
    local jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "  $jobname → $jid  (qos=$qos)"
}

echo "Submitting consolidated AutoResearch round=$ROUND_IDX for ${#CELLS[@]} cells..."
for spec in "${CELLS[@]}"; do
    read -r arch d <<<"$spec"
    # D=5k cells fit in fast queue (4h cap). D=100k needs more time.
    if [ "$d" -ge 50000 ]; then
        submit_one "$arch" "$d" "default" "10:00:00" 4
    else
        submit_one "$arch" "$d" "default" "06:00:00" 4
    fi
done

echo ""
echo "Submitted. Each cell runs 15 configs / 4-parallel ≈ 4 sequential batches."
