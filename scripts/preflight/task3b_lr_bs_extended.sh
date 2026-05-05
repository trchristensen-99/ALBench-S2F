#!/bin/bash
# Pre-flight Task 3b: LR×BS sweep at multiple D values + boundary-edge
# extensions per arch. Catches two failure modes the original Task 3
# missed:
#   1. Locked HP at the GRID EDGE — one step beyond may be better
#      (DREAM-ATTN bs=128 → try bs=64; DRNN bs=256 → try bs=128;
#       LegNet lr=1e-3 → try lr=5e-4).
#   2. SCALE COUPLING — optimum shifts with D. We check at D ∈
#      {500, 6000, 60000} in addition to the locked-D 600k sweep.
#
# Per-arch EXTENDED grid (adds the boundary-edge BS/LR not in original grid):
#   LegNet:    LR [5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2]  × BS [256, 512, 1024]
#   DRNN:      LR [3e-4, 6e-4, 1e-3, 3e-3, 1e-2]        × BS [128, 256, 512, 1024]
#   DREAM-ATTN: LR [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]       × BS [64, 128, 256, 512]
#
# D values: {500, 6000, 60000, 600000}
# Skips cells already covered by results/preflight/task3_lr_bs/
# (D=600k inner grid). Skipping uses result.json existence.
#
# Compute envelope:
#   D=500     fast queue: ~30 sec each
#   D=6k      fast queue: ~2 min each
#   D=60k     default:    ~16 min each
#   D=600k    slow_nice:  ~4h each (only the EDGE cells need running)
#
# Estimated total: ~80 GPU-hours; ~4h wall on 20 slots once slow_nice frees.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

SEED=42
SWEEP=lr_bs_extended
EPOCHS=80   # published default (same as Task 3); will use locked epoch_budget if available

# Pull locked LR/BS so we know which cells are "edge" extensions
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

# Per-arch extended grids
declare -A ARCH_LRS=(
    [legnet]="5e-4 1e-3 3e-3 5e-3 1e-2 3e-2"
    [dream_rnn]="3e-4 6e-4 1e-3 3e-3 1e-2"
    [dream_attn]="1e-4 3e-4 1e-3 3e-3 1e-2"
)
declare -A ARCH_BSS=(
    [legnet]="256 512 1024"
    [dream_rnn]="128 256 512 1024"
    [dream_attn]="64 128 256 512"
)

# D-specific QOS routing (small D → fast, big D → slow_nice)
choose_qos() {
    local d=$1
    if [ "$d" -le 1000 ]; then echo "fast 04:00:00"
    elif [ "$d" -le 10000 ]; then echo "fast 04:00:00"
    elif [ "$d" -le 100000 ]; then echo "default 12:00:00"
    else echo "slow_nice 24:00:00"
    fi
}

# D values for the multi-scale sweep
DS=(500 6000 60000 600000)

n_submitted=0
n_skipped=0
for d in "${DS[@]}"; do
    qos_time=$(choose_qos "$d")
    qos="${qos_time% *}"
    t="${qos_time#* }"
    for arch in legnet dream_rnn dream_attn; do
        for lr in ${ARCH_LRS[$arch]}; do
            for bs in ${ARCH_BSS[$arch]}; do
                # Path layout matches task3_lr_bs so analyze_hp_flatness picks both up
                if [ "$d" -eq 600000 ]; then
                    out="results/preflight/task3_lr_bs/${arch}/lr${lr}_bs${bs}/seed${SEED}"
                else
                    out="results/preflight/task3b_lr_bs_d${d}/${arch}/lr${lr}_bs${bs}/seed${SEED}"
                fi
                if [ -f "${out}/result.json" ]; then
                    n_skipped=$((n_skipped + 1))
                    continue
                fi
                jname="pf3b_${arch}_lr${lr}_bs${bs}_d${d}_s${SEED}"
                if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
                    | grep -qx "$jname"; then continue; fi
                PREFLIGHT_QOS=$qos PREFLIGHT_TIME=$t \
                PREFLIGHT_EPOCHS=$EPOCHS PREFLIGHT_SWEEP=$SWEEP \
                PREFLIGHT_LABEL_SOURCE=ag_oracle \
                PREFLIGHT_OUT=$out \
                    bash scripts/preflight/launch.sh "$arch" "$d" "$SEED" \
                        "lr=$lr" "batch_size=$bs"
                n_submitted=$((n_submitted + 1))
            done
        done
    done
done

echo
echo "=== Task 3b: submitted $n_submitted new runs (skipped $n_skipped already-done) ==="
echo "After all D's complete, run analyze_hp_flatness_multid.py for per-D heatmaps."
