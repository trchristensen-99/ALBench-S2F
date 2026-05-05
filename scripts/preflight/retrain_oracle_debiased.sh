#!/bin/bash
# Pre-staged Option C: retrain a 10-fold AG-S2 oracle ensemble with
# the best debiasing variant from outputs/oracle_neg_sweep/pareto_results.json.
#
# Default variant: creative_i2gc02_scale2x — pareto-best balance of
#   id_r=0.933 (-0.001), ood_r=0.714 (-0.06), random_mean=+0.015 (-98%).
# Override with VARIANT env var to use a different config.
#
# Compute estimate:
#   - 10 fold trainings × ~1.5h on H100 (s2c-style, similar to existing
#     ensemble trained at outputs/stage2_k562_oracle/)
#   - = ~15 GPU-hours; ~1.5h wall on slow_nice 20-slot cap
# Plus pseudolabel inference on the new pool: ~1h additional.
#
# DO NOT FIRE until pre_flight_decisions.yaml is signed off (otherwise
# pre-flight Tasks 5/6/7 are still running with the current oracle's
# pseudolabels and would be invalidated by a swap).

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

VARIANT="${VARIANT:-creative_i2gc02_scale2x}"
OUT_DIR="$REPO/outputs/stage2_k562_oracle_debiased_${VARIANT}"

# Sign-off gate
SIGNOFF=$(uv run --no-sync python -c "
import yaml
print(yaml.safe_load(open('$REPO/results/preflight/pre_flight_decisions.yaml')).get('signoff', {}).get('date') or '')
")
if [ -z "$SIGNOFF" ] && [ "${ALLOW_PRE_SIGNOFF:-0}" != "1" ]; then
    echo "ERROR: pre-flight not signed off. Override with ALLOW_PRE_SIGNOFF=1 if intentional."
    exit 1
fi

# The variant config lives under configs/oracle_neg_sweep/. We assume
# outputs/oracle_neg_sweep/<variant>/fold_0/ has the trained-once
# checkpoint as a baseline; we re-train the FULL 10 folds with the same
# debias config using the s2c per-arch HPs from pre_flight_decisions.yaml.
mkdir -p "$OUT_DIR"

cat > /tmp/_pf_oracle_debias.sh <<EOF
#!/bin/bash
#SBATCH --job-name=pf_oracle_debias_${VARIANT}
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
uv run --no-sync python experiments/train_stage2_k562_hashfrag.py \\
    --config-name stage2_k562_oracle \\
    ++fold_id="\${FOLD}" \\
    ++n_folds=10 \\
    ++stage1_dir="\${S1_DIR}" \\
    ++output_dir="\${OUT_FOLD}" \\
    ++use_full_dataset=True \\
    ++debias_mode=$VARIANT \\
    ++wandb_mode=offline
EOF

JOB=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable /tmp/_pf_oracle_debias.sh)
rm -f /tmp/_pf_oracle_debias.sh
echo "Submitted debiased oracle retrain as array job $JOB"
echo "Output: $OUT_DIR/fold_{0..9}/"
echo "After completion, run scripts/preflight/infer_s2_fold.py with --oracle_dir=$OUT_DIR"
echo "to regenerate pseudolabels."
