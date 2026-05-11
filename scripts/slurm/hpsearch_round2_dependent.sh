#!/bin/bash
# Submit Round 2 HP search jobs as SLURM dependents of Round 1.
# Each new job starts AFTER its corresponding Round 1 job completes (any
# state — success or failure). Round 2 explores the EXTENDED HP space:
# adds block_class (eff/plain/ag) + optimizer (adam/adamw).
#
# Speed tweaks vs Round 1:
#   max_epochs       60 → 40
#   patience         15 → 10
#   n_trials         50 → 30
#   k_parallel        4 → 6    (LegNet ~2-7M params, more fits on V100)
#   ASHA grace_period 8 → 5
#   ASHA max_t       60 → 40
#
# Total wall budget cut roughly 50%. Trial coverage still ample
# (3 strategies × 30 trials × extended space).

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

# (strategy, arch, D_TRAIN, parent_jobid)
declare -a JOBS=(
    "random   legnet     5000    2172233"
    "optuna   legnet     5000    2172234"
    "pbt      legnet     5000    2172237"
    "random   legnet     100000  2172238"
    "optuna   legnet     100000  2172239"
    "pbt      legnet     100000  2172242"
    "random   dream_attn 5000    2172254"
    "random   dream_attn 100000  2172255"
    "optuna   dream_attn 5000    2172256"
    "optuna   dream_attn 100000  2172257"
    "pbt      dream_attn 5000    2172262"
    "pbt      dream_attn 100000  2172263"
)

submit_one() {
    local strategy=$1
    local arch=$2
    local d=$3
    local parent_jid=$4

    local outdir="results/preflight/hpsearch/${strategy}_${arch}_d${d}_r2"
    mkdir -p "$outdir"
    local jobname="hps2_${strategy:0:3}_${arch:0:3}_d${d}"

    # Per-arch / per-D timelimits (tighter than Round 1)
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
export N_TRIALS=30
export MAX_EPOCHS=40
export PATIENCE=10
export GPUS=1
export TRIALS_PER_GPU=6

bash $REPO/scripts/slurm/hpsearch_raytune_cell.sh
EOF
    local jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "  $jobname (dep on $parent_jid) → $jid"
}

echo "Submitting Round 2 (extended HP space) as dependents of Round 1..."
for spec in "${JOBS[@]}"; do
    read -r strategy arch d parent <<<"$spec"
    submit_one "$strategy" "$arch" "$d" "$parent"
done

echo ""
echo "Submitted ${#JOBS[@]} dependent jobs. Each will start when its Round 1 parent finishes."
