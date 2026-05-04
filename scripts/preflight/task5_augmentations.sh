#!/bin/bash
# Pre-flight Task 5: augmentation tests at D=600k, 2 seeds, all 3 archs × 4 augs.
# Decision rule (per spec):
#   - rev-complement: lock ON if it strictly improves over none for all 3 archs
#   - shift: lock ON if strictly improves over rev-complement-only
#   - EvoAug: lock ON if strictly improves over rev-complement+shift
# Otherwise: ablate at D=max in week 6/22.
#
# Total: 4 augs × 3 archs × 2 seeds = 24 runs at D=600k. Each ~12-13h on H100,
# so ~24×12 = 288 GPU-hrs serially. With queue parallelism, ~12-24h wall.
#
# DEPENDENCIES: Tasks 3 (LR, BS) and 4 (epoch budget) must be locked first.
# Reads locked HPs from results/preflight/pre_flight_decisions.yaml.

set -euo pipefail

D_TRAIN=600000
SEEDS=(42 123)
SWEEP=augmentations
AUGS=(none rev_complement rc_shift rc_shift_evoaug)

declare -A ARCH_QOS=( [legnet]=slow_nice [dream_rnn]=default [dream_attn]=slow_nice )
declare -A ARCH_TIME=( [legnet]=24:00:00 [dream_rnn]=12:00:00 [dream_attn]=24:00:00 )

# Load locked LR/BS from pre_flight_decisions.yaml if present; fall back to priors.
DECISIONS=results/preflight/pre_flight_decisions.yaml
if [ ! -f "$DECISIONS" ]; then
    echo "ERROR: $DECISIONS not found. Run after Tasks 3+4 lock HPs."
    exit 1
fi

# Helper: extract a per-arch HP value (locked) from yaml; falls back to ARCH_PRIORS via run_single
get_locked() {
    local arch=$1 field=$2
    uv run --no-sync python -c "
import yaml, sys
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
    [ "$LR" = "NULL" ] || [ "$BS" = "NULL" ] || [ "$EPOCHS" = "NULL" ] && {
        echo "  ERROR: locked HPs missing for $arch (lr=$LR bs=$BS epochs=$EPOCHS); skipping"
        continue
    }
    for aug in "${AUGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            out="results/preflight/task5_augmentations/${arch}/${aug}/seed${seed}"
            if [ -f "${out}/result.json" ]; then continue; fi
            jname="pf5_${arch}_${aug}_s${seed}"
            if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
                | grep -qx "$jname"; then continue; fi
            PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$t \
            PREFLIGHT_EPOCHS=$EPOCHS PREFLIGHT_SWEEP=$SWEEP \
            PREFLIGHT_AUG=$aug PREFLIGHT_LABEL_SOURCE=ag_oracle \
            PREFLIGHT_OUT=$out \
                bash scripts/preflight/launch.sh "$arch" "$D_TRAIN" "$seed" \
                    "lr=$LR" "batch_size=$BS"
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo "=== Task 5: submitted ${n_submitted} new runs (sweep=${SWEEP}) ==="
