#!/bin/bash
# Single-fold debias sweep: probe the design space cheaply (15 configs ×
# 1 fold each ≈ 8-10 GPU-hr) before committing to a full 10-fold ensemble.
# Each cell is a fold-0 retrain warm-started from the existing S1 head.
#
# Routes to fast (priority 4000) + default (priority 1000) queues since
# AG-S2 training takes ~30 min per fold, and these jobs don't need
# slow_nice's 30-day endurance. With 2 fast + 4 default concurrent slots,
# the full sweep wall clock is ~2-3 hr.
#
# RC + shift (±15 bp) augmentation is on by default
# (max_shift=15 in stage2_k562_oracle.yaml).
#
# After all 15 fold-0 jobs land, eval_debias_candidates.py scores each
# on real-label panels (in-dist test, OOD designed, SNV) AND on
# negative-control panels (random DNA at 7 GC levels, dinuc-shuffled).
# Picks the best config; ONLY THEN do we extend to a full 10-fold
# ensemble via promote_winner_to_10fold.sh. This sequence avoids the
# prior failure mode of jumping straight to 10-fold on configs that
# turned out to break OOD.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

VARIANT_TAG="${VARIANT_TAG:-debias_sweep_v1}"
OUT_BASE="$REPO/outputs/oracle_neg_sweep/$VARIANT_TAG"
NEG_DIR="$REPO/data/synthetic_negatives"
NEG_CPG_DIR="$REPO/data/synthetic_negatives_cpg_aware"

# Sign-off gate (skipped when ALLOW_PRE_SIGNOFF=1; currently set to allow
# debias work in parallel with LegNet/DREAM-RNN/DREAM-ATTN pre-flight).
SIGNOFF=$(uv run --no-sync python -c "
import yaml
print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')
")
if [ -z "$SIGNOFF" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1 if intentional."
    exit 1
fi

mkdir -p "$OUT_BASE"

# Each config = "label|hydra_overrides". 15 configs spanning the axes
# prior work flagged as relevant. Negative-augmentation paths point to
# pre-existing TSVs under data/synthetic_negatives* (no regeneration
# needed — they already cover GC-matched, dinuc-shuffle, and CpG-aware
# variants).
CONFIGS=(
  "c00_baseline|"
  "c01_neg20_random|++negatives_path=$NEG_DIR/random_negatives.tsv ++neg_fraction=0.20"
  "c02_neg30_random|++negatives_path=$NEG_DIR/random_negatives.tsv ++neg_fraction=0.30"
  "c03_neg30_gcmatched|++negatives_path=$NEG_DIR/gc_matched_negatives.tsv ++neg_fraction=0.30"
  "c04_neg30_dinuc|++negatives_path=$NEG_DIR/dinuc_shuffled_negatives.tsv ++neg_fraction=0.30"
  "c05_neg30_cpgmixed|++negatives_path=$NEG_CPG_DIR/cpg_mixed.tsv ++neg_fraction=0.30"
  "c06_neg30_highcpg|++negatives_path=$NEG_CPG_DIR/high_cpg_only.tsv ++neg_fraction=0.30"
  "c07_cpginv_low|++debias_mode=cpg_invariance ++debias_lambda=0.02"
  "c08_cpginv_med|++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c09_cpginv_high|++debias_mode=cpg_invariance ++debias_lambda=0.10"
  "c10_spectral|++debias_mode=spectral ++debias_lambda=0.05"
  "c11_groupdro|++debias_mode=group_dro ++debias_lambda=0.10"
  "c12_neg30gc_cpginv|++negatives_path=$NEG_DIR/gc_matched_negatives.tsv ++neg_fraction=0.30 ++debias_mode=cpg_invariance ++debias_lambda=0.05"
  "c13_neg30gc_blocks|++negatives_path=$NEG_DIR/gc_matched_negatives.tsv ++neg_fraction=0.30 ++unfreeze_encoder_blocks=[0,1,2,3,4,5]"
  "c14_neg30gc_blocks_cpginv|++negatives_path=$NEG_DIR/gc_matched_negatives.tsv ++neg_fraction=0.30 ++unfreeze_encoder_blocks=[0,1,2,3,4,5] ++debias_mode=cpg_invariance ++debias_lambda=0.05"
)

# Queue routing: 2 jobs to fast (priority 4000, 4h max) for the
# baseline + first neg-aug variant, the rest to default (priority 1000,
# 12h max, group cap of 20 H100 shared across all users).
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

    # First 2 → fast, rest → default
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
#SBATCH --job-name=pf_debias_$label
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
    echo "  submitted $label as $JOB on qos=$QOS time=$TIME"
    n_submitted=$((n_submitted + 1))
    i=$((i + 1))
done

echo
echo "=== Submitted $n_submitted single-fold debias configs ==="
echo "Output: $OUT_BASE/<label>/fold_0/"
echo "Queues: 2 fast + 13 default. With ~30 min/run and 6 concurrent, ETA ~2-3 hr."
echo
echo "Two-step flow (avoids the prior 'jump straight to 10-fold' failure mode):"
echo "  1. wait for all 15 fold-0 jobs to finish, then:"
echo "     uv run --no-sync python scripts/preflight/eval_debias_candidates.py \\"
echo "         --base $OUT_BASE \\"
echo "         --include_baseline $REPO/outputs/stage2_k562_oracle/fold_0/best_model/checkpoint"
echo
echo "  2. inspect $OUT_BASE/eval_summary.csv; for the top-1 (or top-2) config,"
echo "     extend to 10-fold via promote_winner_to_10fold.sh."
