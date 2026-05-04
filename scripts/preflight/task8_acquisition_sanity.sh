#!/bin/bash
# Pre-flight Task 8: acquisition method sanity check.
# For each non-random method already implemented, run ONE acquisition cycle:
#   - D_init = 600,000 (full pool)
#   - D_acquired = 4,000 sequences
#   - 2 seeds per method
# Confirm:
#   1. Method runs without errors
#   2. Jaccard distance to random selection > 0.3 (selecting different sequences)
#
# Runs on CPU only (acquire_one_cycle.py uses k-mer features and reservoir
# samplers — no GPU needed). 9 methods × 2 seeds = 18 jobs, each ~1 min, on
# the cpu_fill queue (low priority but free).

set -euo pipefail

D_INIT=600000
D_ACQUIRED=4000
SEEDS=(42 123)
REPO_ROOT=/grid/wsbs/home_norepl/christen/ALBench-S2F

# Methods supported by scripts/preflight/acquire_one_cycle.py.
# Reservoir-based: prm_5, prm_20, motif_grammar, gc_matched, dinuc_shuffle
# Model-based proxies (k-mer): uncertainty_ensemble, uncertainty_mc_dropout,
#   diversity_kmeans, diversity_max_distance
METHODS=(
    uncertainty_ensemble uncertainty_mc_dropout
    diversity_kmeans diversity_max_distance
    prm_5 prm_20 motif_grammar gc_matched dinuc_shuffle
)

n_submitted=0
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="${REPO_ROOT}/results/preflight/task8_acquisition_sanity/${method}/seed${seed}"
        if [ -f "${out}/jaccard.json" ]; then continue; fi
        jname="pf8_${method}_s${seed}"
        if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
            | grep -qx "$jname"; then continue; fi
        sbatch_script=$(mktemp)
        cat > "$sbatch_script" <<EOF
#!/bin/bash
#SBATCH --job-name=${jname}
#SBATCH --output=${REPO_ROOT}/logs/%x-%j.out
#SBATCH --error=${REPO_ROOT}/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpu_fill
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=32G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd ${REPO_ROOT} || exit 1
export PYTHONPATH="\$PWD"
mkdir -p ${out}
uv run --no-sync python scripts/preflight/acquire_one_cycle.py \\
    --method ${method} \\
    --d_init ${D_INIT} \\
    --d_acquired ${D_ACQUIRED} \\
    --seed ${seed} \\
    --output_dir ${out}
EOF
        /cm/shared/apps/slurm/current/bin/sbatch "$sbatch_script" || true
        rm -f "$sbatch_script"
        n_submitted=$((n_submitted + 1))
    done
done

echo "=== Task 8: submitted ${n_submitted} acquisition sanity runs ==="
echo "Each run takes ~1 min on CPU. Total wall time ~5-10 min depending on cpu_fill queue."
echo "After all complete, run: scripts/preflight/analyze_task8_acquisition.py"
