#!/bin/bash
# Task 4 verification: long-epoch (240ep) parallel run to validate the
# plateau detection in the primary task4_epoch_budget runs.
#
# Why this exists: yesterday's task4 jobs got preempted on slow_nice and
# saved last.pt files. Today's resubmission resumed from those last.pt
# files — but with our new --early_stop_patience=15, runs whose
# best_epoch was deep in the preempted trajectory triggered patience
# almost immediately on resume. LegNet in particular only trained 1 new
# epoch (gpu_hrs=0.05) before stopping at epoch 34 with best_ep=19.
#
# This verification:
#   - Uses a SEPARATE output dir (no last.pt to resume from → fresh run)
#   - 240 epochs (3x published default), patience=30 (lenient), so it can
#     run further if there are improvements past epoch 19/72/68.
#   - 3 archs × 2 seeds = 6 jobs on slow_nice (long-running OK).
#
# After both primary task4 and this verify complete, compare:
#   - If verify plateau ≈ primary plateau → primary results trustworthy.
#   - If verify plateau >> primary plateau → use verify, lock larger budget.
#
# Routes to slow_nice (priority 100, 30-day max) since these are
# multi-hour runs and we already have new checkpoint-resume logic.

set -euo pipefail

D_TRAIN=600000
SEEDS=(42 123)
SWEEP=epoch_budget_verify
EPOCHS=240
EARLY_STOP_PATIENCE=30   # lenient — let plateau detection have room

declare -A ARCH_QOS=( [legnet]=slow_nice [dream_rnn]=slow_nice [dream_attn]=slow_nice )
declare -A ARCH_TIME=( [legnet]=24:00:00 [dream_rnn]=24:00:00 [dream_attn]=24:00:00 )

DECISIONS=results/preflight/pre_flight_decisions.yaml
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
        echo "  ERROR: locked LR/BS missing for $arch; skipping"
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        out="results/preflight/task4_epoch_budget_verify/${arch}/seed${seed}"
        if [ -f "${out}/result.json" ]; then
            echo "  [skip] ${arch}/seed${seed} — done"
            continue
        fi
        jname="pf_t4v_${arch}_s${seed}"
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
echo "=== Task 4 verify: submitted ${n_submitted} 240-epoch runs ==="
echo "Output: results/preflight/task4_epoch_budget_verify/<arch>/seed<seed>/"
echo "After completion, compare plateau epochs vs the primary task4 results."
