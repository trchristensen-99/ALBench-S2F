#!/bin/bash
# Pre-flight Task 4: epoch-budget calibration via plateau detection.
# Train each arch at D=600k with a 120-epoch cap and early-stop patience
# of 15. The first 240-epoch attempt was preempted on slow_nice without
# saving result.json (older run_single.py lacked checkpoint resume).
# 120 epochs gives plenty of headroom — the preempted runs hit best at
# 19 (legnet), 68 (dream_attn), 72 (dream_rnn) — and patience=15 lets
# each arch stop at its own dataset-size-appropriate plateau rather
# than burning fixed compute. analyze_task4_epoch_budget.py still
# detects the plateau from the saved val curve and locks budget at
# 1.5× plateau per arch.
#
# 6 runs total: 3 archs × 2 seeds (42, 123). On slow_nice with the
# preemption-resilient run_single.py (last.pt resume), each completes
# within a 24h window even with multiple preemptions.
#
# DEPENDENCIES: Task 3 (LR×BS) must be locked via lock_task3_decisions.py.

set -euo pipefail

D_TRAIN=600000
SEEDS=(42 123)
SWEEP=epoch_budget
EPOCHS=120
EARLY_STOP_PATIENCE=15

declare -A ARCH_QOS=( [legnet]=slow_nice [dream_rnn]=slow_nice [dream_attn]=slow_nice )
declare -A ARCH_TIME=( [legnet]=24:00:00 [dream_rnn]=24:00:00 [dream_attn]=24:00:00 )

DECISIONS=results/preflight/pre_flight_decisions.yaml
if [ ! -f "$DECISIONS" ]; then
    echo "ERROR: $DECISIONS not found. Run after Task 3 locks LR/BS via lock_task3_decisions.py."
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
    qos=${ARCH_QOS[$arch]}; t=${ARCH_TIME[$arch]}
    LR=$(get_locked $arch learning_rate)
    BS=$(get_locked $arch batch_size)
    if [ "$LR" = "NULL" ] || [ "$BS" = "NULL" ]; then
        echo "  ERROR: locked LR/BS missing for $arch (lr=$LR bs=$BS); skipping"
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        out="results/preflight/task4_epoch_budget/${arch}/seed${seed}"
        if [ -f "${out}/result.json" ]; then
            echo "  [skip] ${arch}/seed${seed} — done"
            continue
        fi
        jname="pf_t4_${arch}_s${seed}"   # distinct from task6's pf_<arch>_d600000_s<seed>
        if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
            | grep -qx "$jname"; then
            echo "  [skip] $jname already in queue"
            continue
        fi
        PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$t \
        PREFLIGHT_EPOCHS=$EPOCHS \
        PREFLIGHT_EARLY_STOP_PATIENCE=$EARLY_STOP_PATIENCE \
        PREFLIGHT_SWEEP=$SWEEP \
        PREFLIGHT_LABEL_SOURCE=ag_oracle \
        PREFLIGHT_OUT=$out \
        PREFLIGHT_JOB_NAME=$jname \
            bash scripts/preflight/launch.sh "$arch" "$D_TRAIN" "$seed" \
                "lr=$LR" "batch_size=$BS"
        n_submitted=$((n_submitted + 1))
    done
done

echo
echo "=== Task 4 epoch budget: submitted ${n_submitted} long runs (sweep=${SWEEP}) ==="
echo "120-epoch cap, early_stop_patience=${EARLY_STOP_PATIENCE} → each arch stops at its own plateau."
echo "After all 6 runs complete:"
echo "  uv run --no-sync python scripts/preflight/analyze_task4_epoch_budget.py"
echo "  to apply the plateau-then-1.5× rule and lock epoch_budget per arch."
