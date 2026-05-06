#!/bin/bash
# Pre-flight Task 3 RETRY — corrected augmentation regime + extended grids.
#
# Why this is needed (from analyze_preflight_universality.py):
#   - LegNet: original task3 used rev_complement aug, but task5 found
#     LegNet performs WORST with augmentation (none = 0.165 vs rc_shift
#     = 0.318 — nearly 2x worse). Original best LR (5e-4) was at the
#     lower edge of the grid. Re-search WITH aug=none AND a broader
#     LR grid extended downward.
#   - dream_attn: original task3 used rev_complement, but task5 found
#     rc_shift performs better (0.162 vs 0.183). Original best BS (128)
#     was at the lower edge. Re-search WITH aug=rc_shift AND a BS grid
#     extended downward (BS=64).
#   - dream_rnn: NOT retried — rev_complement was the right aug + HPs
#     are interior already.
#
# Routes across all 3 queues for max throughput. With ~30 min per cell
# and 26 GPU cap, full retry should finish in ~2-3 hr wall.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

D_TRAIN=600000
SEED=42

# LegNet retry: extended LR grid (added 1e-4, 3e-4) + standard BS grid
declare -a LEGNET_CONFIGS=(
    "lr=0.0001 batch_size=256"
    "lr=0.0001 batch_size=512"
    "lr=0.0001 batch_size=1024"
    "lr=0.0003 batch_size=256"
    "lr=0.0003 batch_size=512"
    "lr=0.0003 batch_size=1024"
    "lr=0.0005 batch_size=256"
    "lr=0.0005 batch_size=512"
    "lr=0.0005 batch_size=1024"
    "lr=0.001 batch_size=256"
    "lr=0.001 batch_size=512"
    "lr=0.001 batch_size=1024"
    "lr=0.003 batch_size=256"
    "lr=0.003 batch_size=512"
    "lr=0.003 batch_size=1024"
    "lr=0.01 batch_size=256"
    "lr=0.01 batch_size=512"
    "lr=0.01 batch_size=1024"
)

# dream_attn retry: standard LR grid + extended BS grid (added 64)
declare -a DATTN_CONFIGS=(
    "lr=0.0001 batch_size=64"
    "lr=0.0001 batch_size=128"
    "lr=0.0001 batch_size=256"
    "lr=0.0001 batch_size=512"
    "lr=0.0003 batch_size=64"
    "lr=0.0003 batch_size=128"
    "lr=0.0003 batch_size=256"
    "lr=0.0003 batch_size=512"
    "lr=0.001 batch_size=64"
    "lr=0.001 batch_size=128"
    "lr=0.001 batch_size=256"
    "lr=0.001 batch_size=512"
    "lr=0.003 batch_size=64"
    "lr=0.003 batch_size=128"
    "lr=0.003 batch_size=256"
    "lr=0.003 batch_size=512"
    "lr=0.01 batch_size=64"
    "lr=0.01 batch_size=128"
    "lr=0.01 batch_size=256"
    "lr=0.01 batch_size=512"
)

n_submitted=0
i=0
submit_one() {
    local arch=$1 aug=$2 cfg=$3 outroot=$4 i_idx=$5
    # Pick QoS by index for round-robin distribution: 0,1 → fast; 2-5 → default; 6+ → slow_nice
    local qos time
    if [ "$i_idx" -lt 2 ]; then qos=fast; time=04:00:00
    elif [ "$i_idx" -lt 6 ]; then qos=default; time=06:00:00
    else qos=slow_nice; time=12:00:00
    fi
    # Encode cfg into output path — use literal dot→underscore so each LR is unique
    local lr_val bs_val
    lr_val=$(echo "$cfg" | grep -oE 'lr=[0-9.e\-]+' | cut -d= -f2)
    bs_val=$(echo "$cfg" | grep -oE 'batch_size=[0-9]+' | cut -d= -f2)
    local lrtag="${lr_val//./_}"
    local out="$outroot/lr${lrtag}_bs${bs_val}/seed${SEED}"
    if [ -f "${out}/result.json" ]; then
        echo "  [skip] $out — done"
        return 0
    fi
    local jname="pf_t3r_${arch}_lr${lrtag}_bs${bs_val}_s${SEED}"
    if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
        | grep -qx "$jname"; then
        echo "  [skip] $jname already in queue"
        return 0
    fi
    PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$time \
    PREFLIGHT_EPOCHS=80 \
    PREFLIGHT_EARLY_STOP_PATIENCE=15 \
    PREFLIGHT_AUG=$aug \
    PREFLIGHT_SWEEP=task3_retry \
    PREFLIGHT_LABEL_SOURCE=ag_oracle \
    PREFLIGHT_OUT=$out \
    PREFLIGHT_JOB_NAME=$jname \
        bash scripts/preflight/launch.sh "$arch" "$D_TRAIN" "$SEED" $cfg
    n_submitted=$((n_submitted + 1))
}

echo "=== LegNet retry (aug=none, extended LR down) ==="
for cfg in "${LEGNET_CONFIGS[@]}"; do
    submit_one legnet none "$cfg" "results/preflight/task3_retry_legnet_noaug" "$i"
    i=$((i + 1))
done

echo
echo "=== dream_attn retry (aug=rc_shift, extended BS down to 64) ==="
for cfg in "${DATTN_CONFIGS[@]}"; do
    submit_one dream_attn rc_shift "$cfg" "results/preflight/task3_retry_dream_attn_rcshift" "$i"
    i=$((i + 1))
done

echo
echo "=== Submitted $n_submitted retry cells ==="
echo "Outputs:"
echo "  results/preflight/task3_retry_legnet_noaug/<lr_bs>/seed${SEED}/"
echo "  results/preflight/task3_retry_dream_attn_rcshift/<lr_bs>/seed${SEED}/"
echo
echo "After completion, run analyze_task3 on each retry dir to lock the new HPs."
