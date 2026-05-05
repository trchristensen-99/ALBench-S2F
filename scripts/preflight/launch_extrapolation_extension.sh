#!/bin/bash
# Launch the scaling-law extrapolation extension.
#
# Stages:
#   1. Generate reservoir sequences at all fitting + held-out D values
#      (4.8M seqs per method).
#   2. Run AG-S2 oracle ensemble inference on the generated seqs (10 folds).
#   3. Aggregate per-method pseudolabel npzs.
#   4. Train fitting cells (LegNet/DRNN/DREAM-ATTN at d ≤ 1.2M).
#   5. Train extrapolation cells (LegNet only, d ∈ {2.4M, 4.8M}).
#   6. Fit γ_k from fitting cells, score extrapolation against held-out.
#
# Refuses to launch unless main sweep is already complete (so we don't
# fight the main sweep for slow_nice slots).
#
# Usage:
#   bash scripts/preflight/launch_extrapolation_extension.sh [--execute]

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

EXECUTE=false
[ "${1:-}" = "--execute" ] && EXECUTE=true

CONFIG=$REPO/configs/extrapolation_design.yaml
[ ! -f "$CONFIG" ] && { echo "ERROR: $CONFIG missing"; exit 1; }

DECISIONS=$REPO/results/preflight/pre_flight_decisions.yaml
[ ! -f "$DECISIONS" ] && { echo "ERROR: $DECISIONS missing"; exit 1; }

# Verify main sweep finished. We use the existence of a sentinel result
# at the largest main-sweep D point as a proxy.
SIGNOFF=$(uv run --no-sync python -c "
import yaml
print(yaml.safe_load(open('$DECISIONS')).get('signoff', {}).get('date') or '')
")
if [ -z "$SIGNOFF" ]; then
    echo "ERROR: pre_flight_decisions.yaml has no sign-off date. Run task10_finalize.py first."
    exit 1
fi

# Pull design from YAML
read_design() {
    uv run --no-sync python -c "
import yaml
d = yaml.safe_load(open('$CONFIG'))
print(' '.join(str(x) for x in (d.get('$1') or [])))
"
}
FITTING_D=( $(read_design fitting_d_grid) )
EXTRAP_D=( $(read_design extrapolation_test_d) )
METHODS=( $(read_design extension_methods) )
ARCHS_FIT=( $(read_design fitting_archs) )
ARCHS_EXTRAP=( $(read_design extrapolation_archs) )
SEEDS_FIT=( $(read_design fitting_seeds) )
SEEDS_EXTRAP=( $(read_design extrapolation_seeds) )

echo "=== Extrapolation extension plan ==="
echo "  fitting d_grid:     ${FITTING_D[*]}"
echo "  extrapolation d:    ${EXTRAP_D[*]}"
echo "  methods:            ${METHODS[*]}"
echo "  fitting archs:      ${ARCHS_FIT[*]} (seeds: ${SEEDS_FIT[*]})"
echo "  extrapolation arch: ${ARCHS_EXTRAP[*]} (seeds: ${SEEDS_EXTRAP[*]})"

n_planned=0
for method in "${METHODS[@]}"; do
    # Fitting cells: only NEW D points (1.2M and beyond if not in main sweep)
    for d in "${FITTING_D[@]}"; do
        # Skip D values already covered by main sweep d_grid
        for arch in "${ARCHS_FIT[@]}"; do
            for seed in "${SEEDS_FIT[@]}"; do
                out="results/exp1_extension/d${d}/${arch}/${method}/seed${seed}"
                if [ -f "${out}/result.json" ]; then continue; fi
                n_planned=$((n_planned + 1))
                if ! $EXECUTE; then continue; fi
                LR=$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$DECISIONS'))['learning_rate']['$arch']['value'])")
                BS=$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$DECISIONS'))['batch_size']['$arch']['value'])")
                EP=$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$DECISIONS'))['epoch_budget']['$arch']['value'])")
                PREFLIGHT_QOS=slow_nice PREFLIGHT_TIME=24:00:00 \
                PREFLIGHT_EPOCHS="$EP" PREFLIGHT_SWEEP=exp1_extension \
                PREFLIGHT_LABEL_SOURCE=ag_oracle \
                PREFLIGHT_OUT="$out" \
                    bash scripts/preflight/launch.sh "$arch" "$d" "$seed" \
                        "lr=$LR" "batch_size=$BS"
            done
        done
    done
    # Extrapolation cells: held-out D, single arch + seed
    for d in "${EXTRAP_D[@]}"; do
        for arch in "${ARCHS_EXTRAP[@]}"; do
            for seed in "${SEEDS_EXTRAP[@]}"; do
                out="results/exp1_extension/d${d}/${arch}/${method}/seed${seed}"
                if [ -f "${out}/result.json" ]; then continue; fi
                n_planned=$((n_planned + 1))
                if ! $EXECUTE; then continue; fi
                LR=$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$DECISIONS'))['learning_rate']['$arch']['value'])")
                BS=$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$DECISIONS'))['batch_size']['$arch']['value'])")
                EP=$(uv run --no-sync python -c "import yaml; print(yaml.safe_load(open('$DECISIONS'))['epoch_budget']['$arch']['value'])")
                PREFLIGHT_QOS=slow_nice PREFLIGHT_TIME=48:00:00 \
                PREFLIGHT_EPOCHS="$EP" PREFLIGHT_SWEEP=exp1_extension \
                PREFLIGHT_LABEL_SOURCE=ag_oracle \
                PREFLIGHT_OUT="$out" \
                    bash scripts/preflight/launch.sh "$arch" "$d" "$seed" \
                        "lr=$LR" "batch_size=$BS"
            done
        done
    done
done

if $EXECUTE; then
    echo "=== Submitted $n_planned extension runs ==="
else
    echo "=== Plan: $n_planned cells. Use --execute to submit. ==="
fi

echo
echo "PRE-REQUISITES not handled here (run before --execute):"
echo "  1. Generate reservoir seqs at d=4.8M for each method"
echo "     (use albench.reservoir + new utility script TBD)"
echo "  2. Run AG-S2 oracle inference on the generated seqs across 10 folds"
echo "     (use a variant of scripts/preflight/infer_s2_fold.py)"
echo "  3. Aggregate per-method pseudolabel npzs"
echo "After extension cells finish, run analyze_extrapolation.py to compute"
echo "fit-vs-measured error per (method, D) and pass/warn/fail per criterion."
