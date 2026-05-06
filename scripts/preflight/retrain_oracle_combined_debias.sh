#!/bin/bash
# Train a NEW 10-fold AG-S2 oracle ensemble that combines the best
# debiasing signals from prior work (oracle_neg_sweep/) + lessons from
# our bias eval. The trained ensemble PREDICTS corrected labels
# directly — no post-hoc correction layer needed downstream.
#
# Recipe (per fold):
#   - Negative augmentation: 30% of each minibatch is GC-stratified
#     random DNA (GC ∈ {25, 35, 45, 55, 65, 75}%) labeled 0.0
#   - CpG-invariance loss term: λ=0.05 penalty on
#     correlation(residuals, CpG_freq)
#   - Encoder LR 1e-4, head LR 1e-3 (s2c config)
#   - All encoder blocks unfrozen (per breakthrough hypothesis)
#   - 80 epochs (don't push to 100+ — overcorrects per prior work)
#   - Warm-start from existing S1 head ckpts
#
# Compute: ~1.5h per fold on H100, 10 folds in parallel = ~1.5h wall on
# slow_nice (when not contention-limited). Realistic with current
# cluster: ~5-8h.
#
# DO NOT FIRE until pre_flight_decisions.yaml is signed off (Tasks 5/6/7
# are using the current oracle's pseudolabels). Use ALLOW_PRE_SIGNOFF=1
# to override.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

VARIANT_NAME="${VARIANT_NAME:-combined_debias_v1}"
OUT_DIR="$REPO/outputs/stage2_k562_oracle_${VARIANT_NAME}"

# Sign-off gate
SIGNOFF=$(uv run --no-sync python -c "
import yaml
print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')
")
if [ -z "$SIGNOFF" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1 if intentional."
    exit 1
fi

mkdir -p "$OUT_DIR"

cat > /tmp/_pf_combined_debias.sh <<EOF
#!/bin/bash
#SBATCH --job-name=pf_combined_debias_${VARIANT_NAME}
#SBATCH --output=$REPO/logs/%x-%A-%a.out
#SBATCH --error=$REPO/logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --array=0-9
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd $REPO || exit 1
export PYTHONPATH=\"\$PWD\"
export XLA_FLAGS=\"--xla_gpu_enable_command_buffer=\"
source scripts/slurm/setup_hpc_deps.sh

FOLD=\${SLURM_ARRAY_TASK_ID}
S1_DIR="$REPO/outputs/oracle_full_856k/s1/oracle_\${FOLD}"
OUT_FOLD="$OUT_DIR/fold_\${FOLD}"
if [ -f "\${OUT_FOLD}/test_metrics.json" ]; then
    echo "SKIP: fold \${FOLD} already done"
    exit 0
fi

# Combined debias config:
#   - 30% GC-stratified negative augmentation (label=0)
#   - CpG-invariance loss λ=0.05
#   - All encoder blocks unfrozen
#   - s2c LRs (encoder 1e-4, head 1e-3)
#   - 80 epochs
uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \\
    --config-name stage2_k562_oracle \\
    ++fold_id="\${FOLD}" \\
    ++n_folds=10 \\
    ++stage1_dir="\${S1_DIR}" \\
    ++output_dir="\${OUT_FOLD}" \\
    ++use_full_dataset=True \\
    ++neg_augmentation=gc_stratified \\
    ++neg_fraction=0.30 \\
    ++neg_gc_levels='[0.25,0.35,0.45,0.55,0.65,0.75]' \\
    ++debias_mode=cpg_invariance \\
    ++debias_lambda=0.05 \\
    ++unfreeze_encoder_blocks='[0,1,2,3,4,5]' \\
    ++epochs=80 \\
    ++early_stop_patience=15 \\
    ++wandb_mode=online
EOF

JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable /tmp/_pf_combined_debias.sh)
rm -f /tmp/_pf_combined_debias.sh
echo "Submitted combined-debias retrain as array job $JOB"
echo "Output: $OUT_DIR/fold_{0..9}/"
echo
echo "After completion:"
echo "  1. Run scripts/preflight/infer_s2_fold.py with --oracle_dir=$OUT_DIR"
echo "     (regenerates pseudolabels using the new ensemble)"
echo "  2. Run scripts/preflight/score_oracle_bias.py with the new oracle"
echo "     to verify bias reduction"
echo "  3. Compare to current oracle's bias_eval.json — pick the winner"
