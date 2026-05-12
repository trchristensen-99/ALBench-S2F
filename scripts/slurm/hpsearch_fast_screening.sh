#!/bin/bash
# Fast HP screening at D=500 — designed to finish in <30 min per cell.
#
# Pattern: small-D cascade. Run all strategies × all archs at D=500 first
# (cheap, fast), pick the top HPs at each strategy×arch, then seed the D=5k /
# D=100k searches with those for refinement.
#
# Tunable knobs (all aggressive vs Round 1/2):
#   D_TRAIN         500   (vs 5k/100k)
#   N_TRIALS        20    (vs 30-50)
#   MAX_EPOCHS      20    (vs 40-60; D=500 converges fast)
#   PATIENCE         6    (vs 10-15)
#   TRIALS_PER_GPU   8    (small models fit easily; ~750MB each)
#   EVAL_TEST_EVERY  5
#   CUDNN_BENCHMARK  1
#
# Cells: 2 archs × 5 strategies = 10 SLURM jobs. Each ~10-30 min wall.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

declare -a STRATEGIES=(random optuna pbt nevergrad bayesopt)
declare -a ARCHS=(legnet dream_attn)
D_TRAIN=500
QOS=${1:-slow_nice}  # default to slow_nice; pass "fast" if there's headroom

submit_one() {
    local strategy=$1
    local arch=$2

    local outdir="results/preflight/hpsearch/${strategy}_${arch}_d${D_TRAIN}_screen"
    mkdir -p "$outdir"
    local jobname="hpsc_${strategy:0:3}_${arch:0:3}_d${D_TRAIN}"

    # Even smallish D=500 needs ~10 min worst case + Ray startup
    local timelimit="01:30:00"

    local jobfile=$(mktemp)
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=$timelimit
#SBATCH --mem=120G

export STRATEGY=$strategy
export ARCH=$arch
export D_TRAIN=$D_TRAIN
export OUT_DIR=$outdir
export N_TRIALS=20
export MAX_EPOCHS=20
export PATIENCE=6
export GPUS=1
export TRIALS_PER_GPU=8

# All speedups on (fastest possible iteration)
export CUDNN_BENCHMARK=1
export EVAL_ON_GPU=0    # k=8 trials share GPU; conservative
export EVAL_TEST_EVERY=5
export EVAL_BATCH_MULT=2

bash $REPO/scripts/slurm/hpsearch_raytune_cell.sh
EOF
    local jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "  $jobname → $jid"
}

echo "Submitting D=$D_TRAIN fast screening: 2 archs × 5 strategies = 10 jobs..."
for strategy in "${STRATEGIES[@]}"; do
    for arch in "${ARCHS[@]}"; do
        submit_one "$strategy" "$arch"
    done
done

echo ""
echo "All 10 submitted to qos=$QOS. Expected wall ~10-30 min each once running."
