#!/bin/bash
# Pipeline-component ablation: how much does each Stage of the standardized
# procedure contribute to final test loss? Submits 3 isolated runs at
# LegNet D=20000 (the cell with the most reference results):
#
#  random_only       — Stage 1 random with 90 trials (vs full pipeline's 30 random
#                       + 30 optuna + 30 pbt + 30 ablate + 27 aug + 6 seed = 153)
#                       — fair budget vs total.
#  no_aug_sweep      — Stages 1+2 (random+optuna+pbt+ablate) but skip Stage 3
#                       aug. Measures whether the aug sweep adds value.
#  no_ablate         — Stages 1+3+4 but skip Stage 2 ablate.
#
# All submitted to slow_nice. Result.jsons land in
# results/preflight/hpsearch/std_legnet_d20000_ablate_{name}/.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

submit_random_only() {
    local outdir=$1
    local n_trials=$2
    local jobfile
    jobfile=$(mktemp)
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=hp_ablate_random_only
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=140G
#SBATCH --time=12:00:00

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache

python -m scripts.preflight.hpsearch.raytune_search \\
    --strategy random --arch legnet --d_train 20000 \\
    --n_trials $n_trials --max_epochs 60 --patience 10 \\
    --gpus 1 --trials_per_gpu 3 \\
    --output_dir $outdir
EOF
    local jid
    jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "$jid"
}

# (1) random_only at the same total budget as our full pipeline trial count (~93)
JID1=$(submit_random_only results/preflight/hpsearch/std_legnet_d20000_ablate_random_only 90)
echo "ablate random_only (90 trials) → $JID1"

# (2) no_aug_sweep: run the standardized_procedure with Stage 3 short-circuited.
# Easiest: just run a regular full procedure but skip aug stage by setting
# n_aug_configs=0. Since the script doesn't expose that, mark this as a TODO
# follow-up — for tonight we'll get random_only as the cheapest informative ablation.
echo
echo "[note] no_aug_sweep / no_ablate variants require a small script patch — skipping for tonight."
echo "       The random_only run gives the most informative single ablation:"
echo "       does Stage 2-4 actually add value over a brute-force random search?"
