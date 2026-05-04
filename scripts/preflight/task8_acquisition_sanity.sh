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
# This is a sanity check, not a comparative evaluation. Failures are
# documented in pre_flight_decisions.yaml as "acquisition_sanity_flagged".

set -euo pipefail

D_INIT=600000
D_ACQUIRED=4000
SEEDS=(42 123)

# Method list — these match what's already implemented in
# experiments/exp1_2_acquisition.py and the existing reservoirs.
# Subset that should run on the new ref+alt+boda2 cache.
METHODS=(uncertainty_ensemble uncertainty_mc_dropout diversity_kmeans diversity_max_distance prm_5 prm_20 motif_grammar gc_matched dinuc_shuffle)

# This task uses an existing acquisition driver (not run_single.py). Wire to
# experiments/exp1_2_acquisition.py with appropriate flags. Each run is
# small (just acquisition + Jaccard check, no training), so fast queue is fine.
# TIME limit: 2h is plenty for an acquisition cycle without training.

n_submitted=0
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="results/preflight/task8_acquisition_sanity/${method}/seed${seed}"
        if [ -f "${out}/jaccard.json" ]; then continue; fi
        jname="pf8_${method}_s${seed}"
        if /cm/shared/apps/slurm/current/bin/squeue -u christen --noheader -o '%j' \
            | grep -qx "$jname"; then continue; fi
        sbatch_script=$(mktemp)
        cat > "$sbatch_script" <<EOF
#!/bin/bash
#SBATCH --job-name=${jname}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
set -euo pipefail
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5
cd /grid/wsbs/home_norepl/christen/ALBench-S2F || exit 1
export PYTHONPATH="\$PWD"
source scripts/slurm/setup_hpc_deps.sh

# Run one acquisition cycle from the new K562 ref+alt cache.
# Saves selected sequence indices to ${out}/selected_idx.npy and a Jaccard
# distance to a random baseline at ${out}/jaccard.json.
mkdir -p ${out}
uv run --no-sync python scripts/preflight/acquire_one_cycle.py \
    --method ${method} \
    --d_init ${D_INIT} \
    --d_acquired ${D_ACQUIRED} \
    --seed ${seed} \
    --output_dir ${out}
EOF
        /cm/shared/apps/slurm/current/bin/sbatch "$sbatch_script" || true
        rm -f "$sbatch_script"
        n_submitted=$((n_submitted + 1))
    done
done

echo "=== Task 8: submitted ${n_submitted} acquisition sanity runs ==="
echo "(NOTE: scripts/preflight/acquire_one_cycle.py is a placeholder —"
echo " write it after Tasks 3-4 lock so it can use the right student/HPs)"
