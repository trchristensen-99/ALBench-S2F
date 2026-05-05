#!/bin/bash
# Launches the Exp 1.1 main scaling-law sweep using HPs from the locked
# pre-flight YAML. Refuses to launch if pre_flight_decisions.yaml hasn't
# been signed off via task10_finalize.py.
#
# Sweep structure:
#   - 3 archs × N_methods × |d_grid| × N_seeds runs at D_init=0
#   - 3 archs × N_methods × |d_grid| × N_seeds runs at D_init=600000
#
# Default config:
#   - d_grid: from YAML d_grid (set this with task10_finalize.py before launching)
#   - seeds: 3 per cell (bumped to 5 for the headline panel if compute permits)
#   - methods: methods_at_d_init_0 + methods_at_d_init_600k from YAML
#
# This script does NOT submit any jobs — it builds the run plan and prints
# it (or, with --execute, actually submits). Treat it as a generator; the
# real launch should happen after a human dry-run review of the plan.

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

EXECUTE=false
SEEDS=(42 123 7)
DEFAULT_D_GRID=(500 1000 4000 10000 60000 300000 600000)

while [ "$#" -gt 0 ]; do
    case "$1" in
        --execute) EXECUTE=true; shift ;;
        --seeds) shift; IFS=',' read -ra SEEDS <<< "$1"; shift ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

DECISIONS=results/preflight/pre_flight_decisions.yaml
[ ! -f "$DECISIONS" ] && { echo "ERROR: $DECISIONS missing"; exit 1; }

# Validate sign-off
SIGNOFF_DATE=$(uv run --no-sync python -c "import yaml; d = yaml.safe_load(open('$DECISIONS')); print(d.get('signoff', {}).get('date') or 'NULL')")
if [ "$SIGNOFF_DATE" = "NULL" ] || [ -z "$SIGNOFF_DATE" ]; then
    echo "ERROR: pre_flight_decisions.yaml has no sign-off date."
    echo "Run: uv run --no-sync python scripts/preflight/task10_finalize.py --reviewer YOUR_NAME"
    exit 1
fi
echo "Pre-flight YAML signed off on $SIGNOFF_DATE."

# Pull locked HPs + sweep design from YAML
read_yaml() {
    uv run --no-sync python -c "
import yaml
d = yaml.safe_load(open('$DECISIONS'))
def get(p):
    cur = d
    for k in p.split('.'):
        if cur is None: return ''
        cur = cur.get(k)
    return cur if cur is not None else ''
print(get('$1'))
"
}

# d_grid: prefer YAML, fall back to default
D_GRID_FROM_YAML=$(uv run --no-sync python -c "
import yaml
d = yaml.safe_load(open('$DECISIONS'))
g = d.get('d_grid') or []
print(' '.join(str(x) for x in g))
")
if [ -n "$D_GRID_FROM_YAML" ]; then
    read -ra D_GRID <<< "$D_GRID_FROM_YAML"
else
    D_GRID=("${DEFAULT_D_GRID[@]}")
    echo "  d_grid from YAML is empty; using default: ${D_GRID[*]}"
fi

# Filter d_grid by the d_min convergence floor.
# d_min.confirmed (Task 9, val_R²>0.1 across all archs × seeds at locked HPs)
# is the hard floor — any d_grid point below this would be a wasted main-
# sweep cell because the model can't even minimally fit the data there.
# We fall back to d_min.provisional (Task 2) if Task 9 hasn't run.
D_MIN_FLOOR=$(uv run --no-sync python -c "
import yaml
d = yaml.safe_load(open('$DECISIONS')).get('d_min', {})
floor = d.get('confirmed') or d.get('provisional') or 0
print(int(floor))
")
if [ "$D_MIN_FLOOR" -gt 0 ]; then
    FILTERED_GRID=()
    DROPPED=()
    for d in "${D_GRID[@]}"; do
        if [ "$d" -ge "$D_MIN_FLOOR" ]; then
            FILTERED_GRID+=("$d")
        else
            DROPPED+=("$d")
        fi
    done
    if [ ${#DROPPED[@]} -gt 0 ]; then
        echo "  ⚠ Dropping d_grid points below d_min floor ($D_MIN_FLOOR): ${DROPPED[*]}"
        echo "    These points are below the convergence threshold (val_R²>0.1)."
        echo "    Pre-flight Task 2/9 confirmed models can't reliably learn at smaller D."
    fi
    if [ ${#FILTERED_GRID[@]} -eq 0 ]; then
        echo "ERROR: all d_grid points are below d_min floor $D_MIN_FLOOR. Refusing to launch."
        exit 1
    fi
    D_GRID=("${FILTERED_GRID[@]}")
    echo "  d_grid after floor: ${D_GRID[*]}"
fi

D_INIT_VALUES=(0 600000)

# Methods: from YAML lists
methods_d0=$(uv run --no-sync python -c "
import yaml
d = yaml.safe_load(open('$DECISIONS'))
print(' '.join(d.get('methods_at_d_init_0') or []))
")
methods_d600k=$(uv run --no-sync python -c "
import yaml
d = yaml.safe_load(open('$DECISIONS'))
print(' '.join(d.get('methods_at_d_init_600k') or []))
")
if [ -z "$methods_d0" ]; then
    methods_d0="random gc_matched dinuc_shuffle prm_5pct prm_20pct"
    echo "  methods_at_d_init_0 from YAML is empty; using defaults"
fi
if [ -z "$methods_d600k" ]; then
    methods_d600k="$methods_d0 uncertainty_ensemble diversity_max_distance"
    echo "  methods_at_d_init_600k from YAML is empty; using defaults"
fi

n_planned=0
for d_init in "${D_INIT_VALUES[@]}"; do
    if [ "$d_init" = "0" ]; then
        method_list=$methods_d0
    else
        method_list=$methods_d600k
    fi
    for arch in legnet dream_rnn dream_attn; do
        LR=$(read_yaml "learning_rate.$arch.value")
        BS=$(read_yaml "batch_size.$arch.value")
        EP=$(read_yaml "epoch_budget.$arch.value")
        DP_KEY="dropout"
        DP=$(read_yaml "dropout.$arch.value")
        for method in $method_list; do
            for d in "${D_GRID[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    n_planned=$((n_planned + 1))
                    if ! $EXECUTE; then continue; fi
                    out="results/exp1_1/d_init${d_init}/${arch}/${method}/d${d}/seed${seed}"
                    [ -f "${out}/result.json" ] && continue
                    PREFLIGHT_QOS=slow_nice PREFLIGHT_TIME=24:00:00 \
                    PREFLIGHT_EPOCHS="$EP" PREFLIGHT_SWEEP=exp1_1_main \
                    PREFLIGHT_LABEL_SOURCE=ag_oracle \
                    PREFLIGHT_OUT="$out" \
                        bash scripts/preflight/launch.sh "$arch" "$d" "$seed" \
                            "lr=$LR" "batch_size=$BS"
                done
            done
        done
    done
done

if $EXECUTE; then
    echo "=== Main sweep launched: ~${n_planned} runs ==="
else
    echo "=== Plan: ${n_planned} runs would be submitted. Use --execute to submit. ==="
fi
