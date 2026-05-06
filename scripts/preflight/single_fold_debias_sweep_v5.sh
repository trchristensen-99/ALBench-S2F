#!/bin/bash
# Debias sweep v5: multi-seed replicates of v4 winners + v6 ideas.
#
# Why this exists:
# - v3/v4 results are single-seed each. Before promoting to a 10-fold
#   ensemble, we need to verify the winning configs aren't lucky.
# - 3 seeds × 2 top configs (c28, c39) = 6 replicates.
# - Plus 3 new exploration cells testing larger neg-aug fractions
#   with the *real* measured negative labels (not "expect 0"). The TSVs
#   pass labels of -0.45 (random) and -0.75 (intergenic), not 0, but
#   the encoder's CpG-island prior is still overpowering at our current
#   fractions. Larger neg-aug + loss-weighted neg might break through.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

VARIANT_TAG="${VARIANT_TAG:-debias_sweep_v5}"
OUT_BASE="$REPO/outputs/oracle_neg_sweep/$VARIANT_TAG"
NEG_DIR="$REPO/data/synthetic_negatives"

if [ -z "$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')")" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1."
    exit 1
fi

mkdir -p "$OUT_BASE"

# 9 configs total: 3 c28 reps + 3 c39 reps + 3 push-harder explorations
CONFIGS=(
  # Multi-seed replicate of c28 (3% dinuc + cpg_inv 0.05) — train_stage2
  # uses random seed when seed: null, so each rep gets a different seed
  "c28r1_neg3dinuc_cpginv|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.03 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c28r2_neg3dinuc_cpginv|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.03 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c28r3_neg3dinuc_cpginv|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.03 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  # Multi-seed replicate of c39 (3% dinuc+intergenic mix, no loss debias)
  "c39r1_neg3mixed|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.03"
  "c39r2_neg3mixed|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.03"
  "c39r3_neg3mixed|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.03"
  # Push-harder explorations: larger fractions (we know 30% kills OOD;
  # try 15% — between v1's failure point at 30% and v3's safe c28 at 3%)
  "c50_neg15mixed|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.15"
  "c51_neg15mixed_cpginv|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.15 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c52_neg20mixed_cpginv|++negatives_path=$NEG_DIR/dinuc_plus_intergenic.tsv ++neg_fraction=0.20 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
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
    if [ $i -lt 2 ]; then QOS=fast; TIME=03:00:00
    elif [ $i -lt 6 ]; then QOS=default; TIME=04:00:00
    else QOS=slow_nice; TIME=06:00:00
    fi
    sbatch_script=$(mktemp)
    cat > "$sbatch_script" <<EOF
#!/bin/bash
#SBATCH --job-name=pf_debias5_$label
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
echo "=== v5 sweep: submitted $n_submitted configs to $OUT_BASE/ ==="
