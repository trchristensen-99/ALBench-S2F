#!/bin/bash
# Pre-flight Task 4: epoch-budget calibration via plateau detection.
# Per checklist: train ONE D=600k run per arch for 3× the published-default
# epoch count, identify the plateau epoch (no val improvement >0.5% over 10
# consecutive epochs), and lock the budget at 1.5× plateau.
#
# Total: 3 runs (one per arch). Each is the longest training of pre-flight,
# so they're routed to slow_nice with a comfortable time limit.
#
# DEPENDENCIES: Task 3 (LR×BS) must be locked via lock_task3_decisions.py.
# After all 3 runs land, run analyze_task4_epoch_budget.py to apply the
# plateau-then-1.5× rule and write epoch_budget.<arch> into the YAML.
#
# Published-default epoch counts (from priors table):
#   LegNet 80, DREAM-RNN 80, DREAM-ATTN 80 → 3× = 240 each.

set -euo pipefail

D_TRAIN=600000
SEED=42
SWEEP=epoch_budget
EPOCHS=240   # 3× published-default 80 epochs

declare -A ARCH_QOS=( [legnet]=slow_nice [dream_rnn]=slow_nice [dream_attn]=slow_nice )
declare -A ARCH_TIME=( [legnet]=48:00:00 [dream_rnn]=24:00:00 [dream_attn]=48:00:00 )

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
    out="results/preflight/task4_epoch_budget/${arch}/seed${SEED}"
    if [ -f "${out}/result.json" ]; then
        echo "  [skip] ${arch} — done"
        continue
    fi
    jname="pf4_${arch}_ep${EPOCHS}_s${SEED}"
    if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
        | grep -qx "$jname"; then
        echo "  [skip] $jname already in queue"
        continue
    fi
    PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$t \
    PREFLIGHT_EPOCHS=$EPOCHS PREFLIGHT_SWEEP=$SWEEP \
    PREFLIGHT_LABEL_SOURCE=ag_oracle \
    PREFLIGHT_OUT=$out \
        bash scripts/preflight/launch.sh "$arch" "$D_TRAIN" "$SEED" \
            "lr=$LR" "batch_size=$BS"
    n_submitted=$((n_submitted + 1))
done

echo
echo "=== Task 4 epoch budget: submitted ${n_submitted} long runs (sweep=${SWEEP}) ==="
echo "Each run is 3× the published-default budget (240 epochs at D=600k)."
echo "After all 3 runs complete:"
echo "  uv run --no-sync python scripts/preflight/analyze_task4_epoch_budget.py"
echo "  to apply the plateau-then-1.5× rule and lock epoch_budget per arch."
