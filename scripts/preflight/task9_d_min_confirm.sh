#!/bin/bash
# Pre-flight Task 9: D_min CONFIRMATION (locked HPs).
# Re-run the Task 2 D_min sweep, but using the LR / batch_size / epoch_budget
# locked in pre_flight_decisions.yaml from Tasks 3+4. Confirms that the
# provisional D_min from Task 2 (default HPs) holds with the actual main-
# sweep HPs.
#
# Total: 3 archs × 4 D ∈ {500, 1000, 2000, 4000} × 3 seeds = 36 runs.
# Same QOS routing as Task 2; runs are small (≤ 4000 train samples), so
# the fast queue carries most of the load.
#
# DEPENDENCIES: Tasks 3 and 4 must be locked.

set -euo pipefail

D_GRID=(500 1000 2000 4000)
SEEDS=(42 123 7)
SWEEP=d_min_confirm

declare -A ARCH_QOS=( [legnet]=fast [dream_rnn]=default [dream_attn]=fast )

DECISIONS=results/preflight/pre_flight_decisions.yaml
if [ ! -f "$DECISIONS" ]; then
    echo "ERROR: $DECISIONS not found. Run after Tasks 3+4 lock HPs."
    exit 1
fi

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
    qos=${ARCH_QOS[$arch]}
    LR=$(get_locked $arch learning_rate)
    BS=$(get_locked $arch batch_size)
    EPOCHS=$(get_locked $arch epoch_budget)
    if [ "$LR" = "NULL" ] || [ "$BS" = "NULL" ] || [ "$EPOCHS" = "NULL" ]; then
        echo "  ERROR: locked HPs missing for $arch (lr=$LR bs=$BS epochs=$EPOCHS); skipping"
        continue
    fi
    for d in "${D_GRID[@]}"; do
        for seed in "${SEEDS[@]}"; do
            out="results/preflight/task9_d_min_confirm/${arch}/d${d}/seed${seed}"
            if [ -f "${out}/result.json" ]; then continue; fi
            jname="pf9_${arch}_d${d}_s${seed}"
            if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
                | grep -qx "$jname"; then continue; fi
            PREFLIGHT_QOS=$qos PREFLIGHT_TIME=04:00:00 \
            PREFLIGHT_EPOCHS=$EPOCHS PREFLIGHT_SWEEP=$SWEEP \
            PREFLIGHT_LABEL_SOURCE=ag_oracle \
            PREFLIGHT_OUT=$out \
                bash scripts/preflight/launch.sh "$arch" "$d" "$seed" \
                    "lr=$LR" "batch_size=$BS"
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo
echo "=== Task 9 D_min confirm: submitted ${n_submitted} new runs ==="
