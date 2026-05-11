#!/bin/bash
# Launch the full HP search: 6 cells × 5 strategies = 30 SLURM jobs.
#
# Cells: {legnet, dream_rnn, dream_attn} × {5000, 100000}.
# Strategies: random / optuna / hyperopt / bohb / pbt (all with ASHA).
#
# Resource plan:
#   - 1 SLURM job per (arch, D, strategy) — 1 GPU each (V100 ok, H100 better)
#   - Inside, Ray runs 4-8 concurrent trials/GPU (LegNet tiny, DREAM-RNN/ATTN ~5M)
#   - n_trials=50 per strategy at D=5k (fast) and 30 at D=100k (slower)
#   - max_epochs=60 with patience=15. ASHA aborts most trials < 20 epochs.
#
# Queue strategy:
#   - First 16 jobs → slow_nice (cap 20, 30d). LegNet+DREAM-RNN at both D.
#   - Next 14 jobs → default (cap 8). DREAM-ATTN + fillers.
# Adjust if queues are saturated.
#
# Usage: bash scripts/slurm/hpsearch_launch_all.sh

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

declare -a CELLS=(
    "legnet 5000"
    "legnet 100000"
    "dream_rnn 5000"
    "dream_rnn 100000"
    "dream_attn 5000"
    "dream_attn 100000"
)
declare -a STRATEGIES=(random optuna hyperopt bohb pbt)

submit_one() {
    local arch=$1
    local d=$2
    local strategy=$3
    local qos=$4
    local timelimit=$5

    local cell="${arch}_d${d}"
    local outdir="results/preflight/hpsearch/${strategy}_${cell}"
    mkdir -p "$outdir"

    # Per-arch time scaling — DREAM-ATTN ~2-3x slower than LegNet
    local jobname="hps_${strategy}_${arch:0:3}_d${d}"

    local job=$(mktemp)
    cat > "$job" <<EOF
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

export STRATEGY=$strategy
export ARCH=$arch
export D_TRAIN=$d
export OUT_DIR=$outdir
export N_TRIALS=50
export MAX_EPOCHS=60
export PATIENCE=15
export GPUS=1
export TRIALS_PER_GPU=4

bash $REPO/scripts/slurm/hpsearch_raytune_cell.sh
EOF

    local jid=$($SBATCH --parsable "$job")
    rm -f "$job"
    echo "  $jobname → job $jid (qos=$qos)"
}

echo "Submitting 30 jobs (6 cells × 5 strategies)..."
i=0
for cell_spec in "${CELLS[@]}"; do
    read -r arch d <<<"$cell_spec"
    # DREAM-ATTN is heavier — give it longer time
    if [ "$arch" = "dream_attn" ]; then
        timelimit="24:00:00"
    elif [ "$d" -ge 100000 ]; then
        timelimit="24:00:00"
    else
        timelimit="12:00:00"
    fi
    for strategy in "${STRATEGIES[@]}"; do
        # Spread across queues: 20 on slow_nice, rest on default
        if [ "$i" -lt 20 ]; then
            qos="slow_nice"
        else
            qos="default"
        fi
        submit_one "$arch" "$d" "$strategy" "$qos" "$timelimit"
        i=$((i+1))
    done
done

echo ""
echo "Submitted $i jobs total. Use squeue to monitor."
