#!/bin/bash
# Pre-flight Task 3 (verify half): joint-optimum verification at D_min.
# After lock_task3_decisions.py picks the (LR*, BS*) optimum per arch from
# the D_max sweep, re-run that cell PLUS its 2 nearby LR cells at
# D=D_min_provisional with 2 seeds to confirm the optimum is stable across
# scales. This is the "scale coupling" check from the checklist:
#
#   "lock (LR*, BS*) per architecture if joint optimum at D_min and D_max
#    are within one grid step on each axis"
#
# Total: 3 archs × 3 LR (optimum + 2 neighbors) × 1 BS (locked) × 2 seeds
#      = 18 runs at D_min (very fast, fast queue).
#
# DEPENDENCIES: Task 3 D_max + lock_task3_decisions.py must have written
# learning_rate / batch_size into pre_flight_decisions.yaml.

set -euo pipefail

D_TRAIN=500   # D_min_provisional from Task 2
SEEDS=(42 123)
SWEEP=lr_bs_dmin_verify
EPOCHS=80

# Each arch's full LR grid from priors — used to find the 2 LR neighbors
# adjacent to the locked optimum on each axis.
declare -A ARCH_LRS=(
    [legnet]="1e-3 3e-3 5e-3 1e-2 3e-2"
    [dream_rnn]="3e-4 6e-4 1e-3 3e-3 1e-2"
    [dream_attn]="1e-4 3e-4 1e-3 3e-3 1e-2"
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

# Given the locked LR and the full LR grid, return the locked + 2 neighbors.
get_lr_triplet() {
    local arch=$1 lr_locked=$2
    local grid="${ARCH_LRS[$arch]}"
    uv run --no-sync python -c "
import sys
grid = '$grid'.split()
locked = float('$lr_locked')
# Sort numerically and find the locked LR's index
g = sorted(grid, key=lambda v: float(v))
floats = [float(v) for v in g]
# Pick the index whose float matches locked the closest
idx = min(range(len(g)), key=lambda i: abs(floats[i] - locked))
neighbors = []
if idx - 1 >= 0:
    neighbors.append(g[idx - 1])
neighbors.append(g[idx])
if idx + 1 < len(g):
    neighbors.append(g[idx + 1])
print(' '.join(neighbors))
"
}

n_submitted=0
for arch in legnet dream_rnn dream_attn; do
    LR=$(get_locked $arch learning_rate)
    BS=$(get_locked $arch batch_size)
    if [ "$LR" = "NULL" ] || [ "$BS" = "NULL" ]; then
        echo "  ERROR: locked LR/BS missing for $arch (lr=$LR bs=$BS); skipping"
        continue
    fi
    LR_TRIPLET=$(get_lr_triplet $arch $LR)
    echo "  [$arch] verifying LR ∈ {$LR_TRIPLET} at BS=$BS, D=$D_TRAIN"
    for lr in $LR_TRIPLET; do
        for seed in "${SEEDS[@]}"; do
            out="results/preflight/task3_verify_dmin/${arch}/lr${lr}_bs${BS}/seed${seed}"
            if [ -f "${out}/result.json" ]; then continue; fi
            jname="pf3v_${arch}_lr${lr}_s${seed}"
            if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
                | grep -qx "$jname"; then continue; fi
            PREFLIGHT_QOS=fast PREFLIGHT_TIME=04:00:00 \
            PREFLIGHT_EPOCHS=$EPOCHS PREFLIGHT_SWEEP=$SWEEP \
            PREFLIGHT_LABEL_SOURCE=ag_oracle \
            PREFLIGHT_OUT=$out \
                bash scripts/preflight/launch.sh "$arch" "$D_TRAIN" "$seed" \
                    "lr=$lr" "batch_size=$BS"
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo
echo "=== Task 3 verify @ D_min: submitted ${n_submitted} runs ==="
echo "After all complete, run analyze_task3_verify.py to confirm scale-stability."
