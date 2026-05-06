#!/bin/bash
# Promote a single-fold debias-sweep winner to a full 10-fold ensemble.
#
# This is the SECOND stage of the debias workflow. It is intentionally
# decoupled from the sweep so we don't commit ~50 GPU-h on a config we
# haven't validated yet.
#
# Inputs (env vars):
#   WINNER_DIR     — path to the winning candidate's fold_0 dir.
#                    Reads <WINNER_DIR>/.hydra/overrides.yaml (or, fallback,
#                    eval_summary.json from the sweep root) to recover the
#                    exact training overrides to replicate across folds 1-9.
#   ENSEMBLE_NAME  — name for the new ensemble dir (default: derived from
#                    WINNER_DIR's basename).
#
# Refuses to fire pre-signoff. Set ALLOW_PRE_SIGNOFF=1 to override.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

if [ -z "${WINNER_DIR:-}" ]; then
    echo "ERROR: set WINNER_DIR to <oracle_neg_sweep>/<candidate>/fold_0"
    echo "Example: WINNER_DIR=$REPO/outputs/oracle_neg_sweep/debias_sweep_v1/c09_neg30_cpginv/fold_0 \\"
    echo "         ENSEMBLE_NAME=stage2_k562_oracle_debias_v1 bash $0"
    exit 1
fi
WINNER_DIR="$(realpath "$WINNER_DIR")"
ENSEMBLE_NAME="${ENSEMBLE_NAME:-$(basename "$(dirname "$WINNER_DIR")")_10fold}"
OUT_DIR="$REPO/outputs/$ENSEMBLE_NAME"

SIGNOFF=$(uv run --no-sync python -c "
import yaml
print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')
")
if [ -z "$SIGNOFF" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1 if intentional."
    exit 1
fi

# Recover the winning overrides from Hydra's saved record.
HYDRA_OV="$WINNER_DIR/.hydra/overrides.yaml"
if [ -f "$HYDRA_OV" ]; then
    OVERRIDES=$(uv run --no-sync python -c "
import sys, yaml
ov = yaml.safe_load(open('$HYDRA_OV'))
print(' '.join([f'++{x}' if not x.startswith(('++','--')) else x for x in ov]))
")
else
    echo "ERROR: no overrides.yaml at $HYDRA_OV"
    echo "       Cannot replicate winner config — bailing."
    exit 1
fi
echo "Winner overrides: $OVERRIDES"

mkdir -p "$OUT_DIR"

# Copy fold_0 from the sweep so we don't waste 1.5 GPU-h re-training it.
mkdir -p "$OUT_DIR/fold_0"
if [ -d "$WINNER_DIR/best_model" ] && [ ! -d "$OUT_DIR/fold_0/best_model" ]; then
    echo "Linking fold_0 from winner: $WINNER_DIR -> $OUT_DIR/fold_0/"
    cp -r "$WINNER_DIR"/* "$OUT_DIR/fold_0/"
fi

cat > /tmp/_pf_promote.sh <<EOF
#!/bin/bash
#SBATCH --job-name=pf_promote_${ENSEMBLE_NAME}
#SBATCH --output=$REPO/logs/%x-%A-%a.out
#SBATCH --error=$REPO/logs/%x-%A-%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --array=1-9
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd $REPO || exit 1
export PYTHONPATH=\"\$PWD\"
export XLA_FLAGS=\"--xla_gpu_enable_command_buffer=\"
source scripts/slurm/setup_hpc_deps.sh

FOLD=\${SLURM_ARRAY_TASK_ID}
S1_DIR=\"$REPO/outputs/oracle_full_856k/s1/oracle_\${FOLD}\"
OUT_FOLD=\"$OUT_DIR/fold_\${FOLD}\"
if [ -f \"\${OUT_FOLD}/test_metrics.json\" ]; then
    echo \"SKIP: fold \${FOLD} already done\"
    exit 0
fi
uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \\
    --config-name stage2_k562_oracle \\
    ++fold_id=\"\${FOLD}\" \\
    ++n_folds=10 \\
    ++stage1_dir=\"\${S1_DIR}\" \\
    ++output_dir=\"\${OUT_FOLD}\" \\
    ++use_full_dataset=True \\
    ++wandb_mode=online \\
    $OVERRIDES
EOF

JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable /tmp/_pf_promote.sh)
rm -f /tmp/_pf_promote.sh
echo "Submitted folds 1-9 as array job $JOB"
echo "Output: $OUT_DIR/fold_{1..9}/  (fold_0 already in place from sweep)"
echo
echo "After completion:"
echo "  1. uv run --no-sync python scripts/preflight/score_oracle_bias.py \\"
echo "       --oracle_dir $OUT_DIR  --out $OUT_DIR/bias_eval.json"
echo "  2. compare $OUT_DIR/bias_eval.json against the current oracle's bias_eval.json"
echo "  3. only if the new ensemble's bias profile is BETTER (in-dist Pearson"
echo "     preserved AND neg-control means closer to 0), promote to main sweep."
