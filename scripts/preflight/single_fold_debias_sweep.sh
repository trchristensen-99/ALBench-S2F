#!/bin/bash
# Single-fold debias sweep: probe the design space cheaply (15 configs ×
# 1 fold each = 15 GPU-hr) before committing to a full 10-fold ensemble.
# Each cell is a fold-0 retrain with a different combination of debias
# levers, warm-started from the existing S1 head ckpt.
#
# After all 15 finish, eval_debias_candidates.py scores each on the
# real-label panels (in-dist test, OOD designed, SNV) AND on the
# negative-control panels (random DNA at 7 GC levels, dinuc-shuffled).
# Picks the best config; ONLY THEN do we extend to a full 10-fold
# ensemble.
#
# This sequence avoids the prior failure mode of jumping straight to
# 10-fold on configs that turned out to break OOD (frac10_elr5,
# creative_i*gc02_scale2x → 7-21% OOD drop).
#
# DOES NOT FIRE pre-signoff. Set ALLOW_PRE_SIGNOFF=1 to override.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

VARIANT_TAG="${VARIANT_TAG:-debias_sweep_v1}"
OUT_BASE="$REPO/outputs/oracle_neg_sweep/$VARIANT_TAG"

SIGNOFF=$(uv run --no-sync python -c "
import yaml
print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')
")
if [ -z "$SIGNOFF" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1 if intentional."
    exit 1
fi

mkdir -p "$OUT_BASE"

# Each config = (label, hydra-overrides string). 15 configs sampling the
# axes that prior work identified as relevant. Keeps OOD-breaking
# extremes off the menu (no >40% neg-aug, no 100+ epochs).
declare -A CONFIGS
CONFIGS[c00_baseline]=""
CONFIGS[c01_neg20gc]="++neg_augmentation=gc_stratified ++neg_fraction=0.20"
CONFIGS[c02_neg30gc]="++neg_augmentation=gc_stratified ++neg_fraction=0.30"
CONFIGS[c03_neg30gc_blocks]="++neg_augmentation=gc_stratified ++neg_fraction=0.30 ++unfreeze_encoder_blocks=[0,1,2,3,4,5]"
CONFIGS[c04_cpginv_low]="++debias_mode=cpg_invariance ++debias_lambda=0.02"
CONFIGS[c05_cpginv_med]="++debias_mode=cpg_invariance ++debias_lambda=0.05"
CONFIGS[c06_cpginv_high]="++debias_mode=cpg_invariance ++debias_lambda=0.10"
CONFIGS[c07_spectral]="++debias_mode=spectral ++debias_lambda=0.05"
CONFIGS[c08_groupdro]="++debias_mode=group_dro ++debias_lambda=0.10"
CONFIGS[c09_neg30_cpginv]="++neg_augmentation=gc_stratified ++neg_fraction=0.30 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
CONFIGS[c10_neg30_spectral]="++neg_augmentation=gc_stratified ++neg_fraction=0.30 ++debias_mode=spectral ++debias_lambda=0.05"
CONFIGS[c11_neg30_groupdro]="++neg_augmentation=gc_stratified ++neg_fraction=0.30 ++debias_mode=group_dro ++debias_lambda=0.10"
CONFIGS[c12_neg30_blocks_cpginv]="++neg_augmentation=gc_stratified ++neg_fraction=0.30 ++unfreeze_encoder_blocks=[0,1,2,3,4,5] ++debias_mode=cpg_invariance ++debias_lambda=0.05"
CONFIGS[c13_dinuc_neg]="++neg_augmentation=dinuc_shuffle ++neg_fraction=0.30"
CONFIGS[c14_mixed_neg]="++neg_augmentation=mixed ++neg_fraction=0.30"

# Submit one job per config, fold_0 only
n_submitted=0
for label in "${!CONFIGS[@]}"; do
    overrides="${CONFIGS[$label]}"
    out_fold="$OUT_BASE/$label/fold_0"
    if [ -f "$out_fold/test_metrics.json" ]; then continue; fi

    sbatch_script=$(mktemp)
    cat > "$sbatch_script" <<EOF
#!/bin/bash
#SBATCH --job-name=pf_debias_$label
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=06:00:00
#SBATCH --mem=200G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd $REPO || exit 1
export PYTHONPATH="\$PWD"
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
source scripts/slurm/setup_hpc_deps.sh
S1_DIR="$REPO/outputs/oracle_full_856k/s1/oracle_0"
uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \\
    --config-name stage2_k562_oracle \\
    ++fold_id=0 \\
    ++n_folds=10 \\
    ++stage1_dir="\$S1_DIR" \\
    ++output_dir="$out_fold" \\
    ++use_full_dataset=True \\
    ++epochs=80 \\
    ++early_stop_patience=15 \\
    ++wandb_mode=online \\
    $overrides
EOF
    /cm/shared/apps/slurm/current/bin/sbatch "$sbatch_script" || true
    rm -f "$sbatch_script"
    n_submitted=$((n_submitted + 1))
done

echo "=== Submitted $n_submitted single-fold debias configs ==="
echo "Output: $OUT_BASE/<label>/fold_0/"
echo
echo "Two-step flow (avoids the prior 'jump straight to 10-fold' failure mode):"
echo "  1. wait for all 15 fold-0 jobs to finish, then:"
echo "     uv run --no-sync python scripts/preflight/eval_debias_candidates.py \\"
echo "         --base $OUT_BASE \\"
echo "         --include_baseline $REPO/outputs/stage2_k562_oracle/fold_0/best_model/checkpoint"
echo
echo "  2. inspect $OUT_BASE/eval_summary.csv; for the top-1 (or top-2) config,"
echo "     extend to 10-fold by re-running this script's sbatch body with"
echo "     --array=1-9 and the chosen overrides — see promote_winner_to_10fold.sh."
