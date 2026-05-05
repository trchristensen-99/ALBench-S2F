#!/bin/bash
# Pre-flight Task 6: parameterization sensitivity per architecture.
# For each architecture, test 3 parameterizations (0.5×, 1×, 2× the published
# channel/hidden-dim/heads default) at D ∈ {D_min_provisional, D_max} with 2
# seeds. The decision is to LOCK at the published default — this is a
# robustness check (does the curve change with size?), not a tuning sweep.
#
# Total: 3 archs × 3 sizes × 2 D × 2 seeds = 36 runs.
#
# DEPENDENCIES: locks (LR, BS, epoch_budget) from Tasks 3+4.
#
# Per-arch parameterization grid (matches the checklist):
#   LegNet:     channel widths — published default vs 0.5× vs 2× (block_sizes
#               × {0.5, 1.0, 2.0})
#   DREAM-RNN:  hidden_dim — published default 320 vs 160 (0.5×) vs 640 (2×)
#               (cnn_filters scales with the same multiplier)
#   DREAM-ATTN: embedding_dim + num_heads — defaults (256, 4) vs (128, 2)
#               vs (512, 8)

set -euo pipefail

SEEDS=(42 123)
SWEEP=parameterization
D_MIN=500   # provisional from Task 2; same value used in Task 9 confirmation
D_MAX=600000

declare -A ARCH_QOS=( [legnet]=slow_nice [dream_rnn]=slow_nice [dream_attn]=slow_nice )
declare -A ARCH_TIME_DMIN=(
    [legnet]=04:00:00
    [dream_rnn]=04:00:00
    [dream_attn]=04:00:00
)
declare -A ARCH_TIME_DMAX=(
    [legnet]=24:00:00
    [dream_rnn]=12:00:00
    [dream_attn]=24:00:00
)

DECISIONS=results/preflight/pre_flight_decisions.yaml
[ ! -f "$DECISIONS" ] && { echo "ERROR: $DECISIONS missing — Tasks 3+4 must lock first."; exit 1; }

get_locked() {
    local arch=$1 field=$2
    uv run --no-sync python -c "
import yaml
d = yaml.safe_load(open('$DECISIONS'))
v = d.get('$field', {}).get('$arch', {}).get('value')
print(v if v is not None else 'NULL')
"
}

# Build the size variants per arch as `size_label: hp_overrides_string`.
# The hp_overrides_string is a sequence of `key=value` tokens passed as
# extra args to launch.sh.
declare -A SIZES_LEGNET=(
    [half]="block_sizes=[128,128,64,64,32,32,16,16]"
    [default]=""   # use ARCH_PRIORS default
    [double]="block_sizes=[512,512,256,256,128,128,64,64]"
)
declare -A SIZES_DRNN=(
    [half]="hidden_dim=160 cnn_filters=80"
    [default]=""
    [double]="hidden_dim=640 cnn_filters=320"
)
declare -A SIZES_DATTN=(
    [half]="embedding_dim=128 num_heads=2"
    [default]=""
    [double]="embedding_dim=512 num_heads=8"
)

declare -A ARCH_SIZE_KEYS=(
    [legnet]="half default double"
    [dream_rnn]="half default double"
    [dream_attn]="half default double"
)

n_submitted=0
for arch in legnet dream_rnn dream_attn; do
    LR=$(get_locked $arch learning_rate)
    BS=$(get_locked $arch batch_size)
    EPOCHS=$(get_locked $arch epoch_budget)
    if [ "$LR" = "NULL" ] || [ "$BS" = "NULL" ]; then
        echo "  ERROR: locked LR/BS missing for $arch (lr=$LR bs=$BS); skipping"
        continue
    fi
    if [ "$EPOCHS" = "NULL" ]; then
        EPOCHS=80
        echo "  [$arch] epoch_budget not yet locked — using published default 80 (provisional)"
    fi
    qos=${ARCH_QOS[$arch]}
    case "$arch" in
        legnet)     declare -n SIZES=SIZES_LEGNET ;;
        dream_rnn)  declare -n SIZES=SIZES_DRNN ;;
        dream_attn) declare -n SIZES=SIZES_DATTN ;;
    esac
    for size_label in ${ARCH_SIZE_KEYS[$arch]}; do
        size_overrides=${SIZES[$size_label]}
        for d in $D_MIN $D_MAX; do
            # D_min runs are tiny (~30 min); D_max runs are full-length.
            # Both go to the arch's primary qos so we don't contend with
            # whatever else is using fast queue. slow_nice has plenty of
            # slots and a 24h time limit handles either D.
            t=${ARCH_TIME_DMAX[$arch]}
            d_qos=$qos
            for seed in "${SEEDS[@]}"; do
                out="results/preflight/task6_parameterization/${arch}/size_${size_label}/d${d}/seed${seed}"
                if [ -f "${out}/result.json" ]; then continue; fi
                jname="pf6_${arch}_${size_label}_d${d}_s${seed}"
                if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
                    | grep -qx "$jname"; then continue; fi
                PREFLIGHT_QOS=$d_qos PREFLIGHT_TIME=$t \
                PREFLIGHT_EPOCHS=$EPOCHS PREFLIGHT_SWEEP=$SWEEP \
                PREFLIGHT_LABEL_SOURCE=ag_oracle \
                PREFLIGHT_OUT=$out \
                    bash scripts/preflight/launch.sh "$arch" "$d" "$seed" \
                        "lr=$LR" "batch_size=$BS" $size_overrides
                n_submitted=$((n_submitted + 1))
            done
        done
    done
done

echo
echo "=== Task 6 parameterization: submitted ${n_submitted} new runs (sweep=${SWEEP}) ==="
echo "Decision: lock published default per arch (robustness check, not tune)."
