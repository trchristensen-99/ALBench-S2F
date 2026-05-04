#!/bin/bash
# Pre-flight Task 2: D_min provisional sweep.
#
# Train each of (legnet, dream_rnn, dream_attn) at D ∈ {500, 1000, 2000, 4000}
# with 3 seeds {42, 123, 7}, using each architecture's published-default HPs
# (= the centers of the sweep grids in configs/preflight/arch/*.yaml). All
# runs use chromosome-based K562 splits and AG-oracle pseudolabels.
#
# Total: 3 × 4 × 3 = 36 runs. All small-D, expected ~5–10 min each on H100.
# Sent to fast queue (4h limit, 2 concurrent slots — drains in ~2-3h end-to-end).
#
# Output dirs follow:
#   results/preflight/task2_d_min/<arch>/d<D>/seed<seed>/result.json
# Sweep tag: sweep=d_min_provisional
#
# Run from the project root:
#   bash scripts/preflight/task2_d_min_provisional.sh

set -euo pipefail

# Per-arch epoch budget for this short-D pre-flight check. Published defaults
# at D=600k are ~80–150; small-D shouldn't need more than 80 to reveal trend.
EPOCHS=80

ARCHS=(legnet dream_rnn dream_attn)
DS=(500 1000 2000 4000)
SEEDS=(42 123 7)
SWEEP=d_min_provisional

echo "=== Pre-flight Task 2: D_min provisional ==="
echo "  archs   = ${ARCHS[*]}"
echo "  d_train = ${DS[*]}"
echo "  seeds   = ${SEEDS[*]}"
echo "  total   = $((${#ARCHS[@]} * ${#DS[@]} * ${#SEEDS[@]})) runs"
echo

n_submitted=0
for arch in "${ARCHS[@]}"; do
    for d in "${DS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            out="results/preflight/task2_d_min/${arch}/d${d}/seed${seed}"
            # Skip if a result already exists (idempotent re-runs)
            if [ -f "${out}/result.json" ]; then
                echo "  [skip] ${arch} d=${d} seed=${seed} — result exists"
                continue
            fi
            PREFLIGHT_QOS=fast \
            PREFLIGHT_TIME=04:00:00 \
            PREFLIGHT_EPOCHS=${EPOCHS} \
            PREFLIGHT_OUT=${out} \
            PREFLIGHT_SWEEP=${SWEEP} \
            PREFLIGHT_LABEL_SOURCE=ag_oracle \
                bash scripts/preflight/launch.sh "${arch}" "${d}" "${seed}"
            n_submitted=$((n_submitted + 1))
        done
    done
done

echo
echo "=== submitted ${n_submitted} new runs (sweep=${SWEEP}) ==="
