#!/bin/bash
# Debias sweep v3: refine the OOD-safe corridor identified by v1+v2.
#
# Key v1+v2 findings (24 configs, real OOD measured):
#   - Loss-only debias (cpg_invariance, spectral, group_dro, gradient_penalty,
#     counterfactual, conditional, adaptive_dro) PRESERVES OOD perfectly but
#     barely shifts the bias. Toothless on its own.
#   - Neg-aug at 30% kills OOD (0.77 -> 0.30-0.43 R, 8-13x worse MSE).
#   - Sweet-spot found: c23 (10% dinuc + cpg_inv lambda=0.05): random_dna
#     +0.09 (vs +0.90 baseline), OOD 0.612 (vs 0.770; 16% drop).
#     c24 (5% dinuc + cpg_inv lambda=0.10): random_dna +0.37, OOD 0.697.
#     c16 (10% dinuc alone): random_dna +0.21, OOD 0.664.
#   - Intergenic prediction is stable +0.85 to +1.06 across EVERY config —
#     untouched even by c25 which used real intergenic seqs as negatives.
#
# v3 strategy: refine the c23/c24 corridor + try mechanisms not yet tested:
#   1. Smaller neg-aug fractions (2%, 3%) + cpg_inv to map the bias/OOD
#      Pareto frontier more finely.
#   2. Stronger cpg_invariance lambda (0.10, 0.20) at moderate neg-aug.
#   3. Different loss formulations COMBINED with 10% neg-aug (each loss
#      type was only tested alone in v2).
#   4. Mixed-type neg-aug (dinuc_plus_intergenic) to see if combining
#      negative signals attacks intergenic bias.
#   5. Head-only training with neg-aug (freeze encoder entirely): does
#      this preserve OOD where encoder fine-tuning + neg-aug fails?
#   6. Very-strong cpg_invariance alone (lambda=0.50) — last-shot test
#      to see if loss-only CAN work with enough strength.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

VARIANT_TAG="${VARIANT_TAG:-debias_sweep_v3}"
OUT_BASE="$REPO/outputs/oracle_neg_sweep/$VARIANT_TAG"
NEG_DIR="$REPO/data/synthetic_negatives"

if [ -z "$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')")" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1."
    exit 1
fi

mkdir -p "$OUT_BASE"

# 12 configs across 6 strategy groups
CONFIGS=(
  # Fine-grained neg-aug + cpg_inv (Pareto frontier mapping)
  "c27_neg2dinuc_cpginv|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.02 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c28_neg3dinuc_cpginv|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.03 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  # Stronger cpg_inv lambda
  "c29_neg5dinuc_cpginv_high|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.05 ++debias_mode=cpg_invariance ++debias_lambda=0.10"
  "c30_neg10dinuc_cpginv_xhigh|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.10 ++debias_mode=cpg_invariance ++debias_lambda=0.20"
  # Other losses + small neg-aug (each loss type only tested alone in v2)
  "c31_neg10dinuc_grad|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.10 ++debias_mode=cpg_gradient_penalty ++debias_lambda=0.05"
  "c32_neg10dinuc_cond|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.10 ++debias_mode=conditional_invariance ++debias_lambda=0.05"
  "c33_neg10dinuc_counter|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.10 ++debias_mode=counterfactual_consistency ++debias_lambda=0.10"
  # Mixed neg-aug types — does combining attack intergenic bias?
  "c34_neg10mixed|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.10"
  "c35_neg10mixed_cpginv|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.10 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  # Head-only (freeze encoder entirely) — does this rescue OOD with strong neg-aug?
  "c36_head_only_neg30dinuc|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.30 ++unfreeze_encoder_blocks=[]"
  "c37_head_only_neg10dinuc_cpginv|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.10 ++debias_mode=cpg_invariance ++debias_lambda=0.05 ++unfreeze_encoder_blocks=[]"
  # Loss-only at extreme lambda — last shot for the "no neg-aug" path
  "c38_cpginv_extreme|++debias_mode=cpg_invariance ++debias_lambda=0.50"
)

n_submitted=0
i=0
for cell in "${CONFIGS[@]}"; do
    label="${cell%%|*}"
    overrides="${cell#*|}"
    out_fold="$OUT_BASE/$label/fold_0"
    if [ -f "$out_fold/test_metrics.json" ]; then
        echo "  [skip] $label — done"
        i=$((i + 1))
        continue
    fi
    if [ $i -lt 2 ]; then QOS=fast; TIME=03:00:00; else QOS=default; TIME=04:00:00; fi
    sbatch_script=$(mktemp)
    cat > "$sbatch_script" <<EOF
#!/bin/bash
#SBATCH --job-name=pf_debias3_$label
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=$TIME
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
    JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable "$sbatch_script") || {
        echo "  FAILED to submit $label"
        rm -f "$sbatch_script"
        i=$((i + 1))
        continue
    }
    rm -f "$sbatch_script"
    echo "  submitted $label as $JOB on qos=$QOS"
    n_submitted=$((n_submitted + 1))
    i=$((i + 1))
done

echo
echo "=== v3 sweep: submitted $n_submitted configs to $OUT_BASE/ ==="
