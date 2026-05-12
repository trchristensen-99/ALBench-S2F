#!/bin/bash
# Round 4: dedicated D=20k Bayesian sweep — the cell that matters most for
# the current Colab. Round 1/2/3 covered D=5k and D=100k; this fills the gap
# at the exact D the notebook defaults to.
#
# Architecture: LegNet only (DREAM-ATTN unlikely to beat at this D + cheap
# to add later if needed). 4 strategies × 1 cell = 4 SLURM jobs.
#
# Dependency: starts after Round 3 D=100k jobs (the heaviest) finish so we
# don't compete for slow_nice slots with active runs.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

# Parents: Round 3 LegNet D=100k jobs (the slowest cell, finishes last)
# 2187126 = nev_legnet_d100000, 2187127 = bay_legnet_d100000
declare -a JOBS=(
    "random   2187126"
    "optuna   2187126"
    "pbt      2187127"
    "bayesopt 2187127"
)

submit_one() {
    local strategy=$1
    local parent_jid=$2
    local d=20000
    local arch=legnet

    local outdir="results/preflight/hpsearch/${strategy}_${arch}_d${d}_r4"
    mkdir -p "$outdir"
    local jobname="hps4_${strategy:0:3}_leg_d${d}"
    local timelimit="06:00:00"

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
export N_TRIALS=40
export MAX_EPOCHS=40
export PATIENCE=10
export GPUS=1
export TRIALS_PER_GPU=6
# HP_FAST=1 picks up all speedups; compile disabled at the run_single layer
# via the same fallback that worked for shootout_d20k_fillin.

bash $REPO/scripts/slurm/hpsearch_raytune_cell.sh
EOF
    local jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "  $jobname (dep on $parent_jid) → $jid"
}

echo "Submitting Round 4 (D=20k focused Bayesian sweep, dependent on Round 3)..."
for spec in "${JOBS[@]}"; do
    read -r strategy parent <<<"$spec"
    submit_one "$strategy" "$parent"
done

echo ""
echo "Submitted ${#JOBS[@]} jobs. Will run after Round 3 LegNet D=100k finishes."
