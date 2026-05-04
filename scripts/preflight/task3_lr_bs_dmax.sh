#!/bin/bash
# Pre-flight Task 3 (D_max half): joint LR×BS sweep at D=600k, one seed per
# (arch, LR, BS) cell. Runs the full grid for each architecture using the
# new ref+alt+boda2 AG-S2 pseudolabel cache.
#
# LegNet:    5 LR × 3 BS = 15
# DREAM-RNN: 5 LR × 3 BS = 15
# DREAM-ATTN: 5 LR × 4 BS = 20  (extra BS=128 for memory)
# Total: 50 runs.
#
# Each run is at D=600k = ~12-13h on H100. Spread across queues to avoid
# per-queue submit limits.

set -euo pipefail

D_TRAIN=600000
SEED=42
SWEEP=lr_bs_dmax
EPOCHS=80

# Per-arch QOS rotation (same as Task 2)
declare -A ARCH_QOS=( [legnet]=fast [dream_rnn]=default [dream_attn]=slow_nice )
declare -A ARCH_TIME=( [legnet]=04:00:00 [dream_rnn]=12:00:00 [dream_attn]=24:00:00 )

# fast queue can't fit 12-13h LegNet runs — bump LegNet to slow_nice for
# this Task 3 D_max sweep specifically (small D was fine on fast).
ARCH_QOS[legnet]=slow_nice
ARCH_TIME[legnet]=24:00:00

# LR×BS grids per arch (centers + neighbors per priors table)
declare -A ARCH_LRS=(
    [legnet]="1e-3 3e-3 5e-3 1e-2 3e-2"
    [dream_rnn]="3e-4 6e-4 1e-3 3e-3 1e-2"
    [dream_attn]="1e-4 3e-4 1e-3 3e-3 1e-2"
)
declare -A ARCH_BSS=(
    [legnet]="256 512 1024"
    [dream_rnn]="256 512 1024"
    [dream_attn]="128 256 512"   # +128 below for the 4-BS DREAM-ATTN grid
)

n_submitted=0
for arch in legnet dream_rnn dream_attn; do
    qos=${ARCH_QOS[$arch]}; t=${ARCH_TIME[$arch]}
    for lr in ${ARCH_LRS[$arch]}; do
        for bs in ${ARCH_BSS[$arch]}; do
            out="results/preflight/task3_lr_bs/${arch}/lr${lr}_bs${bs}/seed${SEED}"
            if [ -f "${out}/result.json" ]; then
                echo "  [skip] ${arch} lr=${lr} bs=${bs} — done"
                continue
            fi
            jname="pf3_${arch}_lr${lr}_bs${bs}_s${SEED}"
            if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
                | grep -qx "$jname"; then
                echo "  [skip] $jname already in queue"
                continue
            fi
            PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$t \
            PREFLIGHT_EPOCHS=$EPOCHS PREFLIGHT_SWEEP=$SWEEP \
            PREFLIGHT_LABEL_SOURCE=ag_oracle \
            PREFLIGHT_OUT=$out \
                bash scripts/preflight/launch.sh "$arch" "$D_TRAIN" "$SEED" "lr=$lr" "batch_size=$bs"
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo
echo "=== Task 3 D_max: submitted ${n_submitted} new runs (sweep=${SWEEP}) ==="
