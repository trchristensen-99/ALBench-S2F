#!/bin/bash
# Debias sweep v2: targets the gaps revealed by v1.
#
# v1 lessons:
#   - Neg-aug at 30% is a cliff: bias panels improve but OOD R collapses
#     0.77 → 0.33-0.47 (4-13× worse OOD MSE). Useful but unusable.
#   - cpg_invariance λ=0.02 (c07) preserves OOD AND test metrics, but
#     barely moves the bias (random_dna +0.90 → +0.78).
#   - Intergenic prediction (+0.87 to +1.08) is stable across ALL v1
#     configs — none of them touch this bias. Likely a real CpG-island
#     shortcut that needs intergenic-as-negative supervision.
#
# v2 strategy: smaller neg-aug fractions (5-10%) to find the sweet spot,
# four other debias_mode formulations we hadn't tried, combos of small
# neg-aug + loss-based, and a direct attack on intergenic bias.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

VARIANT_TAG="${VARIANT_TAG:-debias_sweep_v2}"
OUT_BASE="$REPO/outputs/oracle_neg_sweep/$VARIANT_TAG"
NEG_DIR="$REPO/data/synthetic_negatives"

if [ -z "$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')")" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1."
    exit 1
fi

mkdir -p "$OUT_BASE"

# 10 configs targeting v1 failure modes:
#   c15-c18: small fractions (5%, 10%) of the neg-aug TSVs that worked
#            best on bias panels in v1 (dinuc, gcmatched).
#   c19-c22: four debias_modes we hadn't tried in v1.
#   c23-c24: small neg-aug + loss-based combos (theory: low neg-aug
#            preserves OOD; loss-based finishes off remaining bias).
CONFIGS=(
  "c15_neg5_dinuc|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.05"
  "c16_neg10_dinuc|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.10"
  "c17_neg5_gcmatched|++negatives_path=$NEG_DIR/gc_matched_negatives.tsv ++neg_fraction=0.05"
  "c18_neg10_gcmatched|++negatives_path=$NEG_DIR/gc_matched_negatives.tsv ++neg_fraction=0.10"
  "c19_grad_penalty|++debias_mode=cpg_gradient_penalty ++debias_lambda=0.05"
  "c20_counter_consist|++debias_mode=counterfactual_consistency ++debias_lambda=0.10"
  "c21_cond_invariance|++debias_mode=conditional_invariance ++debias_lambda=0.05"
  "c22_adaptive_dro|++debias_mode=adaptive_group_dro ++debias_lambda=0.10"
  "c23_neg10dinuc_cpginv|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.10 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c24_neg5dinuc_cpginv|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.05 ++debias_mode=cpg_invariance ++debias_lambda=0.10"
  "c25_neg10intergenic|++negatives_path=$NEG_DIR/real_inter_negative_only.tsv ++neg_fraction=0.10"
  "c26_neg10intergenic_cpginv|++negatives_path=$NEG_DIR/real_inter_negative_only.tsv ++neg_fraction=0.10 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
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

    if [ $i -lt 2 ]; then
        QOS=fast
        TIME=03:00:00
    else
        QOS=default
        TIME=04:00:00
    fi

    sbatch_script=$(mktemp)
    cat > "$sbatch_script" <<EOF
#!/bin/bash
#SBATCH --job-name=pf_debias2_$label
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
echo "=== v2 sweep: submitted $n_submitted configs to $OUT_BASE/ ==="
echo "Score after sweep with: eval_debias_candidates.py --base $OUT_BASE"
