#!/bin/bash
# Pre-flight Task 7: dropout sensitivity at D=600k, 2 seeds, all 3 archs.
# Per arch, test 0.5x, 1x, 2x of published-default dropout.
# Decision: lock at published default (this is robustness check, not a tune).
# Total: 3 archs × 3 dropouts × 2 seeds = 18 runs at D=600k.
#
# DEPENDENCIES: Tasks 3 + 4 locked.
# Per-arch dropout grid is hard-coded below to match plan spec.

set -euo pipefail

D_TRAIN=600000
SEEDS=(42 123)
SWEEP=dropout

declare -A ARCH_QOS=( [legnet]=slow_nice [dream_rnn]=default [dream_attn]=slow_nice )
declare -A ARCH_TIME=( [legnet]=24:00:00 [dream_rnn]=12:00:00 [dream_attn]=24:00:00 )

# Per-arch dropout values (0.5x, default, 2x).
# LegNet: default=0.0; "0.5x of 0" doesn't exist, so use {0.0, 0.1, 0.2}.
# DREAM-RNN: default dropout_lstm=0.3; sweep {0.15, 0.30, 0.50} — clipped to [0,1].
# DREAM-ATTN: default core_dropout=0.1; sweep {0.05, 0.10, 0.20}.
declare -A ARCH_DROPOUT_KEY=(
    [legnet]=dropout
    [dream_rnn]=dropout_lstm
    [dream_attn]=core_dropout
)
declare -A ARCH_DROPOUT_VALUES=(
    [legnet]="0.0 0.1 0.2"           # default 0.0; degenerate at 0.5x — use {0, +0.1, +0.2}
    [dream_rnn]="0.15 0.30 0.60"     # default 0.30: 0.5x=0.15, 2x=0.60
    [dream_attn]="0.05 0.10 0.20"    # default 0.10: 0.5x=0.05, 2x=0.20
)

DECISIONS=results/preflight/pre_flight_decisions.yaml
[ ! -f "$DECISIONS" ] && { echo "ERROR: $DECISIONS missing"; exit 1; }

get_locked() {
    local arch=$1 field=$2
    uv run --no-sync python -c "
import yaml
d = yaml.safe_load(open('$DECISIONS'))
v = d.get('$field', {}).get('$arch', {}).get('value')
print(v if v is not None else 'NULL')
"
}

n_submitted=0
for arch in legnet dream_rnn dream_attn; do
    qos=${ARCH_QOS[$arch]}; t=${ARCH_TIME[$arch]}
    LR=$(get_locked $arch learning_rate)
    BS=$(get_locked $arch batch_size)
    EPOCHS=$(get_locked $arch epoch_budget)
    [ "$LR" = "NULL" ] && continue
    DROPOUT_KEY=${ARCH_DROPOUT_KEY[$arch]}
    for d_val in ${ARCH_DROPOUT_VALUES[$arch]}; do
        for seed in "${SEEDS[@]}"; do
            out="results/preflight/task7_dropout/${arch}/${DROPOUT_KEY}_${d_val}/seed${seed}"
            if [ -f "${out}/result.json" ]; then continue; fi
            jname="pf7_${arch}_${DROPOUT_KEY}${d_val}_s${seed}"
            if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
                | grep -qx "$jname"; then continue; fi
            PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$t \
            PREFLIGHT_EPOCHS=$EPOCHS PREFLIGHT_SWEEP=$SWEEP \
            PREFLIGHT_LABEL_SOURCE=ag_oracle \
            PREFLIGHT_OUT=$out \
                bash scripts/preflight/launch.sh "$arch" "$D_TRAIN" "$seed" \
                    "lr=$LR" "batch_size=$BS" "${DROPOUT_KEY}=$d_val"
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "=== Task 7: submitted ${n_submitted} new runs (sweep=${SWEEP}) ==="
