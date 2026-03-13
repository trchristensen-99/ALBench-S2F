#!/bin/bash
# Experiment 1.1: Full pipeline launcher.
#
# Phase 1 (main experiments, all 6 reservoirs):
#   K562: DREAM-RNN student × AG oracle
#   Yeast: DREAM-RNN student × DREAM-RNN oracle
#
# Phase 2 (oracle architecture comparison, random reservoir only):
#   Train missing oracle ensembles → generate pseudolabels → run 2×2 comparison:
#   K562:  {DREAM-RNN, AG S1} student × {AG, DREAM-RNN} oracle
#   Yeast: {DREAM-RNN, AG S1} student × {DREAM-RNN, AG} oracle
#
# Usage:
#   bash scripts/slurm/launch_exp1_1.sh phase1           # Main experiments
#   bash scripts/slurm/launch_exp1_1.sh phase2-oracles    # Train oracle ensembles
#   bash scripts/slurm/launch_exp1_1.sh phase2-labels     # Generate pseudolabels
#   bash scripts/slurm/launch_exp1_1.sh phase2-comparison # Run 2×2 comparison
#   bash scripts/slurm/launch_exp1_1.sh all               # Everything with deps

set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
mkdir -p logs

SBATCH="/cm/shared/apps/slurm/current/bin/sbatch"
PHASE="${1:-phase1}"

echo "=== Experiment 1.1 Launcher: ${PHASE} ==="

# Helper: submit scaling job for one student×oracle combo on random reservoir only
submit_2x2() {
    local task=$1
    local student=$2
    local oracle=$3
    local dep=${4:-}
    local dep_flag=""
    if [[ -n "${dep}" ]]; then
        dep_flag="--dependency=afterok:${dep}"
    fi
    echo "  ${task} ${student} × ${oracle} oracle (random reservoir)"
    ${SBATCH} --parsable --array=0 ${dep_flag} \
        --export=ALL,TASK=${task},STUDENT=${student},ORACLE=${oracle} \
        scripts/slurm/exp1_1_scaling.sh
}

if [[ "${PHASE}" == "phase1" || "${PHASE}" == "all" ]]; then
    echo ""
    echo "--- Phase 1: Main experiments (all 6 reservoirs) ---"

    echo "K562: DREAM-RNN × AG oracle (default)"
    K562_MAIN=$(TASK=k562 STUDENT=dream_rnn ORACLE=default ${SBATCH} --parsable scripts/slurm/exp1_1_scaling.sh)
    echo "  Job: ${K562_MAIN}"

    echo "Yeast: DREAM-RNN × DREAM-RNN oracle (default)"
    YEAST_MAIN=$(TASK=yeast STUDENT=dream_rnn ORACLE=default ${SBATCH} --parsable scripts/slurm/exp1_1_scaling.sh)
    echo "  Job: ${YEAST_MAIN}"
fi

if [[ "${PHASE}" == "phase2-oracles" || "${PHASE}" == "all" ]]; then
    echo ""
    echo "--- Phase 2a: Train oracle ensembles ---"

    echo "K562 DREAM-RNN oracle (10 folds)"
    K562_DREAM_ORACLE=$(${SBATCH} --parsable scripts/slurm/train_oracle_dream_rnn_k562_ensemble.sh)
    echo "  Job: ${K562_DREAM_ORACLE}"

    echo "Yeast AG oracle (10 folds)"
    YEAST_AG_ORACLE=$(${SBATCH} --parsable scripts/slurm/train_oracle_alphagenome_yeast_ensemble.sh)
    echo "  Job: ${YEAST_AG_ORACLE}"
fi

if [[ "${PHASE}" == "phase2-labels" || "${PHASE}" == "all" ]]; then
    echo ""
    echo "--- Phase 2b: Generate pseudolabels + test set NPZs ---"

    # These depend on oracle training completing
    K562_DREAM_DEP=""
    YEAST_AG_DEP=""
    if [[ "${PHASE}" == "all" ]]; then
        K562_DREAM_DEP="--dependency=afterok:${K562_DREAM_ORACLE}"
        YEAST_AG_DEP="--dependency=afterok:${YEAST_AG_ORACLE}"
    fi

    echo "K562 DREAM-RNN pseudolabels"
    K562_PLABEL=$(${SBATCH} --parsable ${K562_DREAM_DEP} scripts/slurm/generate_k562_dream_pseudolabels.sh)
    echo "  Job: ${K562_PLABEL}"

    echo "Yeast AG pseudolabels"
    YEAST_PLABEL=$(${SBATCH} --parsable ${YEAST_AG_DEP} scripts/slurm/generate_yeast_ag_pseudolabels.sh)
    echo "  Job: ${YEAST_PLABEL}"
fi

if [[ "${PHASE}" == "phase2-comparison" || "${PHASE}" == "all" ]]; then
    echo ""
    echo "--- Phase 2c: 2×2 oracle comparison (random reservoir) ---"

    K562_DEP=""
    YEAST_DEP=""
    if [[ "${PHASE}" == "all" ]]; then
        K562_DEP="${K562_PLABEL}"
        YEAST_DEP="${YEAST_PLABEL}"
    fi

    # K562: 2 students × 2 oracles
    echo "K562 comparisons:"
    submit_2x2 k562 dream_rnn ag          # cross-arch (also in Phase 1, but random only here for speed)
    submit_2x2 k562 dream_rnn dream_rnn "${K562_DEP}"  # same-arch
    submit_2x2 k562 alphagenome_k562_s1 ag              # same-arch
    submit_2x2 k562 alphagenome_k562_s1 dream_rnn "${K562_DEP}"  # cross-arch

    # Yeast: 2 students × 2 oracles
    echo "Yeast comparisons:"
    submit_2x2 yeast dream_rnn dream_rnn        # same-arch (also in Phase 1)
    submit_2x2 yeast dream_rnn ag "${YEAST_DEP}"  # cross-arch
    submit_2x2 yeast alphagenome_yeast_s1 dream_rnn       # cross-arch
    submit_2x2 yeast alphagenome_yeast_s1 ag "${YEAST_DEP}"  # same-arch
fi

echo ""
echo "Check status: squeue -u \$USER"
