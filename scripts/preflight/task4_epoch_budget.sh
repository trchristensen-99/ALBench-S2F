#!/bin/bash
# Pre-flight Task 4: epoch budget calibration at D=600k, locked LR/BS, 1 seed.
# Per arch, sweep ∈ {40, 60, 80, 100} epochs and pick the smallest budget where
# best val MSE plateaus AND best epoch is NOT in the final 10% (per Task 2's
# "min_val_in_final_pct" sanity flag).
#
# Total: 3 archs × 4 budgets = 12 runs at D=600k. Same QOS routing as Task 3.
# Each ~12-13h on H100, so ~12×12 = 144 GPU-hrs serial; with parallelism, ~4-8h
# wall clock.
#
# DEPENDENCIES: Task 3 (LR×BS) must be locked first via lock_task3_decisions.py.

set -euo pipefail

D_TRAIN=600000
SEED=42
SWEEP=epoch_budget
EPOCHS_GRID=(40 60 80 100)

declare -A ARCH_QOS=( [legnet]=slow_nice [dream_rnn]=default [dream_attn]=slow_nice )
declare -A ARCH_TIME=( [legnet]=24:00:00 [dream_rnn]=12:00:00 [dream_attn]=24:00:00 )

DECISIONS=results/preflight/pre_flight_decisions.yaml
if [ ! -f "$DECISIONS" ]; then
    echo "ERROR: $DECISIONS not found. Run after Task 3 locks LR/BS via lock_task3_decisions.py."
    exit 1
fi

# Helper: extract a per-arch HP value (locked) from yaml; falls back to ARCH_PRIORS via run_single
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
    if [ "$LR" = "NULL" ] || [ "$BS" = "NULL" ]; then
        echo "  ERROR: locked LR/BS missing for $arch (lr=$LR bs=$BS); skipping"
        continue
    fi
    for ep in "${EPOCHS_GRID[@]}"; do
        out="results/preflight/task4_epoch_budget/${arch}/ep${ep}/seed${SEED}"
        if [ -f "${out}/result.json" ]; then
            echo "  [skip] ${arch} ep=${ep} — done"
            continue
        fi
        jname="pf4_${arch}_ep${ep}_s${SEED}"
        if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
            | grep -qx "$jname"; then
            echo "  [skip] $jname already in queue"
            continue
        fi
        PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$t \
        PREFLIGHT_EPOCHS=$ep PREFLIGHT_SWEEP=$SWEEP \
        PREFLIGHT_LABEL_SOURCE=ag_oracle \
        PREFLIGHT_OUT=$out \
            bash scripts/preflight/launch.sh "$arch" "$D_TRAIN" "$SEED" \
                "lr=$LR" "batch_size=$BS"
        n_submitted=$((n_submitted + 1))
    done
done

echo
echo "=== Task 4 epoch budget: submitted ${n_submitted} new runs (sweep=${SWEEP}) ==="
echo "After all runs complete, run:"
echo "  uv run --no-sync python scripts/preflight/analyze_task4_epoch_budget.py"
echo "  to lock epoch_budget per arch into pre_flight_decisions.yaml."
