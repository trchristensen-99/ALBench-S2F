#!/bin/bash
# Debias sweep v4: combine the two v3 winning strategies.
#
# v3 results identified two complementary winning directions:
#   c28 (3% dinuc + cpg_inv 0.05): OOD 0.754 (only 1.6% drop) +
#       random_dna 0.90→0.56. Best OOD-bias trade-off across 39 configs.
#   c34 (10% dinuc+intergenic): FIRST config to move intergenic
#       (+0.92 → +0.72). But OOD dropped to 0.67 (13% worse).
#
# v4 hypothesis: low-fraction mixed neg-aug (2-5% dinuc+intergenic) +
# cpg_invariance gets c28's OOD safety AND c34's intergenic touch.
#
# Plus: c28 replicate to confirm single-seed result wasn't lucky.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

VARIANT_TAG="${VARIANT_TAG:-debias_sweep_v4}"
OUT_BASE="$REPO/outputs/oracle_neg_sweep/$VARIANT_TAG"
NEG_DIR="$REPO/data/synthetic_negatives"

if [ -z "$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')")" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1."
    exit 1
fi

mkdir -p "$OUT_BASE"

CONFIGS=(
  # Mixed neg-aug at low fractions (THE main probe — combine c28 OOD safety + c34 intergenic touch)
  "c39_neg3mixed|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.03"
  "c40_neg5mixed|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.05"
  "c41_neg3mixed_cpginv|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.03 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c42_neg5mixed_cpginv|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.05 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c43_neg5mixed_cpginv_high|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.05 ++debias_mode=cpg_invariance ++debias_lambda=0.10"
  # Pure intergenic-only at low fractions — does intergenic alone work without dinuc?
  "c44_neg3intergenic_cpginv|++negatives_path=$NEG_DIR/real_inter_negative_only.tsv ++neg_fraction=0.03 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c45_neg5intergenic_cpginv|++negatives_path=$NEG_DIR/real_inter_negative_only.tsv ++neg_fraction=0.05 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  # Replicate of v3 winner (c28) with default seed → verify single-seed wasn't lucky
  "c46_c28_replicate|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.03 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
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
#SBATCH --job-name=pf_debias4_$label
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
echo "=== v4 sweep: submitted $n_submitted configs to $OUT_BASE/ ==="
