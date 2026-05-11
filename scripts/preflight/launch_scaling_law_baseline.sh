#!/bin/bash
# Launch reservoir-sampling SCALING LAW experiments for the BASELINE oracle
# (no debias) — uses existing labeled pools in outputs/labeled_pools/k562/ag_s2/.
#
# Strategies (Tier 1):
#   random, genomic (capped 600k), prm_5pct, evoaug_structural
# Architectures: LegNet, DREAM-RNN, DREAM-ATTN
# Sizes: 1k, 5k, 10k, 30k, 100k, 300k, 600k, 1M, 2M (genomic only up to 600k)
# Seeds: 3 (42, 123, 456) for empirical CI
#
# Per-arch SLURM array (4 strategies). Each task runs ALL training sizes for
# one (arch, strategy) combo with k_parallel scheduling.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

POOL_DIR="$REPO/outputs/labeled_pools/k562/ag_s2"
OUT_BASE="$REPO/outputs/exp1_1/k562_scaling_baseline_oracle"

mkdir -p "$OUT_BASE"

# Architectures and small/large size tiers
declare -A ARCH_QOS=([legnet]="default" [dream_rnn]="default" [dream_attn]="slow_nice")
declare -A ARCH_GPU=([legnet]="v100" [dream_rnn]="v100" [dream_attn]="h100")
declare -A ARCH_K=([legnet]=4 [dream_rnn]=2 [dream_attn]=3)

STRATEGIES=(random genomic prm_5pct evoaug_structural)
SMALL_SIZES="1000 5000 10000 30000 100000"
LARGE_SIZES_NONGENOMIC="300000 600000 1000000 2000000"
LARGE_SIZES_GENOMIC="300000 500000"  # genomic capped

JOB_IDS=()
for ARCH in legnet dream_rnn dream_attn; do
    QOS=${ARCH_QOS[$ARCH]}
    GPU=${ARCH_GPU[$ARCH]}
    K=${ARCH_K[$ARCH]}
    for i in "${!STRATEGIES[@]}"; do
        STRAT="${STRATEGIES[$i]}"
        if [ "$STRAT" = "genomic" ]; then
            LARGE="$LARGE_SIZES_GENOMIC"
        else
            LARGE="$LARGE_SIZES_NONGENOMIC"
        fi

        SCRIPT=$(mktemp)
        {
            echo "#!/bin/bash"
            echo "#SBATCH --job-name=scl_base_${ARCH}_${STRAT}"
            echo "#SBATCH --output=$REPO/logs/%x-%j.out"
            echo "#SBATCH --error=$REPO/logs/%x-%j.err"
            echo "#SBATCH --partition=gpuq --qos=$QOS --gres=gpu:${GPU}:1 --cpus-per-task=14 --time=12:00:00 --mem=120G"
            echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
            echo "module load EB5; cd $REPO"
            echo 'export PYTHONPATH="$PWD"'
            echo "export TORCHDYNAMO_DISABLE=1"
            echo "source scripts/slurm/setup_hpc_deps.sh"
            echo ""
            echo "# Small tier (k=$K parallel cells per GPU)"
            echo "uv run --no-sync python experiments/exp1_1_scaling.py \\"
            echo "    --task k562 --student $ARCH --oracle ag_s2 \\"
            echo "    --reservoir $STRAT \\"
            echo "    --pool-base-dir $POOL_DIR \\"
            echo "    --n-replicates 3 --seed 42 \\"
            echo "    --output-dir $OUT_BASE/${ARCH} \\"
            echo "    --training-sizes $SMALL_SIZES \\"
            echo "    --chr-split --epochs 80 --ensemble-size 1 --early-stop-patience 10 || true"
            echo ""
            echo "# Large tier (transfer HP from 50k anchor)"
            echo "uv run --no-sync python experiments/exp1_1_scaling.py \\"
            echo "    --task k562 --student $ARCH --oracle ag_s2 \\"
            echo "    --reservoir $STRAT \\"
            echo "    --pool-base-dir $POOL_DIR \\"
            echo "    --n-replicates 3 --seed 42 \\"
            echo "    --output-dir $OUT_BASE/${ARCH} \\"
            echo "    --training-sizes $LARGE \\"
            echo "    --chr-split --epochs 50 --ensemble-size 1 --early-stop-patience 10 \\"
            echo "    --transfer-hp-from 30000 || true"
        } > $SCRIPT
        JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT)
        rm -f $SCRIPT
        echo "  scl_base_${ARCH}_${STRAT}: $JID (qos=$QOS gpu=$GPU)"
        JOB_IDS+=("$JID")
    done
done

echo
echo "=== Submitted ${#JOB_IDS[@]} baseline scaling-law jobs ==="
echo "JOB_IDS: ${JOB_IDS[*]}"
