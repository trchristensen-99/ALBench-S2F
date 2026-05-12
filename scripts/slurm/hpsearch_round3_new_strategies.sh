#!/bin/bash
# Round 3: Nevergrad + BayesOpt searchers, dependent on Round 2 finishing.
# Same speedup defaults as Round 2. 4 cells × 2 strategies = 8 jobs.
#
# Why these strategies:
# - Nevergrad (OnePlusOne): evolutionary, distinct from Random/TPE/PBT;
#   handles small/discrete spaces well.
# - BayesOpt: scikit-optimize-style GP-EI, complements Optuna/HyperOpt's
#   TPE-style Bayesian.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

# (strategy, arch, D_TRAIN, parent_round2_jobid)
declare -a JOBS=(
    "nevergrad legnet     5000    2184103"
    "bayesopt  legnet     5000    2184103"
    "nevergrad legnet     100000  2184106"
    "bayesopt  legnet     100000  2184106"
    "nevergrad dream_attn 5000    2184109"
    "bayesopt  dream_attn 5000    2184109"
    "nevergrad dream_attn 100000  2184110"
    "bayesopt  dream_attn 100000  2184110"
)

submit_one() {
    local strategy=$1
    local arch=$2
    local d=$3
    local parent_jid=$4

    local outdir="results/preflight/hpsearch/${strategy}_${arch}_d${d}_r3"
    mkdir -p "$outdir"
    local jobname="hps3_${strategy:0:3}_${arch:0:3}_d${d}"

    local timelimit
    if [ "$arch" = "dream_attn" ] || [ "$d" -ge 100000 ]; then
        timelimit="08:00:00"
    else
        timelimit="04:00:00"
    fi

    local jobfile=$(mktemp)
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=$timelimit
#SBATCH --mem=140G
#SBATCH --dependency=afterany:$parent_jid

export STRATEGY=$strategy
export ARCH=$arch
export D_TRAIN=$d
export OUT_DIR=$outdir
export N_TRIALS=20
export MAX_EPOCHS=40
export PATIENCE=10
export GPUS=1
export TRIALS_PER_GPU=6
export CUDNN_BENCHMARK=1
export EVAL_TEST_EVERY=5
export EVAL_BATCH_MULT=2

bash $REPO/scripts/slurm/hpsearch_raytune_cell.sh
EOF
    local jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "  $jobname (dep on $parent_jid) → $jid"
}

echo "Submitting Round 3 (nevergrad + bayesopt, dependent on Round 2)..."
for spec in "${JOBS[@]}"; do
    read -r strategy arch d parent <<<"$spec"
    submit_one "$strategy" "$arch" "$d" "$parent"
done

echo ""
echo "Submitted ${#JOBS[@]} jobs. Each starts when its Round 2 parent finishes."
