#!/bin/bash
# Launch comprehensive bias evaluation for all 10-fold oracles.
# Runs RC-averaged predictions on multiple REAL MPRA datasets +
# computes feature-aware bias metrics (CpG slope, GC slope, length bias).

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

OUT_DIR=$REPO/results/preflight/comprehensive_bias
mkdir -p $OUT_DIR

declare -A ORACLES=(
    [baseline]="$REPO/outputs/stage2_k562_oracle"
    [c28_10fold]="$REPO/outputs/oracle_neg_sweep/debias_oracle_c28_10fold"
    [c63_10fold]="$REPO/outputs/oracle_neg_sweep/debias_c63_10fold"
    [c86_10fold]="$REPO/outputs/oracle_neg_sweep/debias_c86_10fold"
    [c91_10fold]="$REPO/outputs/oracle_neg_sweep/debias_c91_10fold"
)

for NAME in "${!ORACLES[@]}"; do
    ORACLE_DIR="${ORACLES[$NAME]}"
    SCRIPT=$(mktemp)
    {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=cbias_${NAME}"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=fast --gres=gpu:h100:1 --cpus-per-task=14 --time=03:00:00 --mem=200G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo "export PYTHONPATH=\"\$PWD\""
    echo "export XLA_FLAGS=--xla_gpu_enable_command_buffer="
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/_comprehensive_bias_eval.py \\"
    echo "    --oracle-name $NAME \\"
    echo "    --oracle-dir $ORACLE_DIR \\"
    echo "    --out-dir $OUT_DIR"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT 2>&1)
    rm -f $SCRIPT
    echo "  cbias_${NAME}: $JID"
done
