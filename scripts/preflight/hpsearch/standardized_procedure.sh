#!/bin/bash
# ============================================================================
# Standardized HP-search procedure for one (ARCH, D_TRAIN) cell.
#
# Stages:
#   1. PARALLEL COARSE SEARCH (3 SLURM jobs in parallel)
#        random + optuna + pbt, each ~30-50 trials in one job
#   2. AUTORESEARCH REFINEMENT (1 SLURM job, depends on Stage 1)
#        2 rounds × 15 LLM-proposed configs (relies on user invoking subagent;
#        falls back to the autoresearch_orchestrator scaffold)
#   3. AUGMENTATION SWEEP (1 SLURM job, depends on Stage 2)
#        Top-3 configs × {rev_complement, rc_shift, rc_shift_evoaug} ×
#        {max_shift ∈ 0, 15, 25} × 1 seed = ~27 configs
#   4. SEED VARIANCE (1 SLURM job, depends on Stage 3)
#        Top-2 of Stage 3 × 3 seeds = 6 final runs
#   5. COVERAGE AUDIT (lightweight CPU job)
#        Verifies every dimension was probed.
#
# Transfer mode:
#   If ANCHOR_DIR is set, Stages 1+2 are SKIPPED and only Stage 3 + Stage 4 run
#   on the top-N configs from the anchor. Use this for {D values × reservoir
#   strategies} that share an arch with a previously-run anchor.
#
# Usage:
#   ARCH=legnet D_TRAIN=20000 bash scripts/preflight/hpsearch/standardized_procedure.sh
#
#   # Transfer mode:
#   ARCH=legnet D_TRAIN=5000 \
#       ANCHOR_DIR=results/preflight/hpsearch/std_legnet_d20000 \
#       bash scripts/preflight/hpsearch/standardized_procedure.sh
# ============================================================================

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

: "${ARCH:?ARCH required (legnet | dream_attn)}"
: "${D_TRAIN:?D_TRAIN required}"
ANCHOR_DIR=${ANCHOR_DIR:-}
N_STAGE1_TRIALS=${N_STAGE1_TRIALS:-30}
N_AUTORESEARCH_ROUNDS=${N_AUTORESEARCH_ROUNDS:-2}
TOP_K_FOR_AUG_SWEEP=${TOP_K_FOR_AUG_SWEEP:-3}
TOP_K_FOR_SEED_VAR=${TOP_K_FOR_SEED_VAR:-2}

OUT_BASE="results/preflight/hpsearch/std_${ARCH}_d${D_TRAIN}"
mkdir -p "$OUT_BASE"
echo "[procedure] $(date) ARCH=$ARCH D_TRAIN=$D_TRAIN  → $OUT_BASE"
echo "[procedure] ANCHOR_DIR=${ANCHOR_DIR:-<none — full procedure>}"

# Common SLURM wrapper params
SLURM_COMMON="--partition=gpuq --qos=default --gres=gpu:1 --cpus-per-task=14 --mem=140G"

# Helper: submit a Ray Tune strategy
submit_raytune() {
    local strategy=$1
    local jobname="std_${ARCH}_d${D_TRAIN}_${strategy}"
    local out="${OUT_BASE}/stage1_${strategy}"
    mkdir -p "$out"
    local jobfile=$(mktemp)
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
$SLURM_COMMON
#SBATCH --time=06:00:00

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache

python -m scripts.preflight.hpsearch.raytune_search \\
    --strategy $strategy --arch $ARCH --d_train $D_TRAIN \\
    --n_trials $N_STAGE1_TRIALS --max_epochs 60 --patience 15 \\
    --gpus 1 --trials_per_gpu 4 \\
    --output_dir $out
EOF
    local jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "$jid"
}

# Helper: submit a config-list runner with optional dependency
submit_configlist() {
    local jobname=$1
    local configs_path=$2
    local k_parallel=$3
    local dep=$4
    local time_lim=${5:-04:00:00}
    local jobfile=$(mktemp)
    local dep_flag=""
    [ -n "$dep" ] && dep_flag="--dependency=afterany:$dep"
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
$SLURM_COMMON
#SBATCH --time=$time_lim

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache
export EVAL_BATCH_MULT=2

uv run --no-sync python -u scripts/preflight/parallel_gpu_runner.py "$configs_path" $k_parallel
EOF
    local jid=$($SBATCH --parsable $dep_flag "$jobfile")
    rm -f "$jobfile"
    echo "$jid"
}

source .venv/bin/activate
S1_DEP=""

if [ -z "$ANCHOR_DIR" ]; then
    # ---- STAGE 1: parallel coarse search ----
    echo "[stage1] submitting random + optuna + pbt in parallel..."
    S1_RANDOM=$(submit_raytune random)
    S1_OPTUNA=$(submit_raytune optuna)
    S1_PBT=$(submit_raytune pbt)
    echo "  stage1 jobs: random=$S1_RANDOM optuna=$S1_OPTUNA pbt=$S1_PBT"
    S1_DEP="afterany:${S1_RANDOM}:${S1_OPTUNA}:${S1_PBT}"
else
    echo "[stage1] SKIPPED (transfer mode from $ANCHOR_DIR)"
fi

# ---- STAGE 2: AutoResearch refinement (placeholder — needs main Claude to drive) ----
# We submit a stub job that prepares the digest; user/orchestrator invokes
# subagents to write proposals; this script does NOT call the LLM.
S2_OUT="${OUT_BASE}/stage2_autoresearch"
mkdir -p "$S2_OUT"
if [ -z "$ANCHOR_DIR" ]; then
    # Digest stage 1 winners (waits for stage 1)
    DIGEST_JOBFILE=$(mktemp)
    cat > "$DIGEST_JOBFILE" <<EOF
#!/bin/bash
#SBATCH --job-name=std_${ARCH}_d${D_TRAIN}_digest
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=cpuq
#SBATCH --qos=cpuq_base
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

cd $REPO
source .venv/bin/activate
python -m scripts.preflight.hpsearch.aggregate_trials
python -m scripts.preflight.hpsearch.coverage_audit --arch $ARCH --d_train $D_TRAIN
echo "[digest] Stage 1 leaderboard for $ARCH D=$D_TRAIN written; ready for AutoResearch."
echo "[digest] Run: python -m scripts.preflight.hpsearch.autoresearch_orchestrator --arch $ARCH --d_train $D_TRAIN"
EOF
    DIGEST_DEP=""
    [ -n "$S1_DEP" ] && DIGEST_DEP="--dependency=$S1_DEP"
    S2_JID=$($SBATCH --parsable $DIGEST_DEP "$DIGEST_JOBFILE")
    rm -f "$DIGEST_JOBFILE"
    echo "[stage2] digest job (AutoResearch trigger) → $S2_JID"
else
    echo "[stage2] using ANCHOR_DIR=$ANCHOR_DIR; skipping coarse search."
fi

# ---- STAGE 3: augmentation sweep on top-K configs from stage 2 (or anchor) ----
# Build aug-sweep configs.json post-hoc once stage 2 / anchor is available.
# This is the same pattern as agent_r4_d20k_launch.sh.
echo "[stage3] augmentation sweep will be assembled after Stage 2 completes."
echo "[stage3]   (top-${TOP_K_FOR_AUG_SWEEP} × {rev_complement, rc_shift+max_shift∈{15,25}, rc_shift_evoaug+intensity∈{1,2,4}})"

# ---- STAGE 4: 3-seed validation on top-2 of stage 3 ----
echo "[stage4] 3-seed validation will run after Stage 3."

# ---- STAGE 5: final coverage audit ----
echo "[stage5] final coverage audit will run after Stage 4."

echo ""
echo "[procedure] Stage 1+2 submitted. Stage 3+4 need to be assembled after Stage 1+2 results land."
echo "[procedure] Next: when Stage 2 digest writes the leaderboard, run:"
echo "[procedure]    bash scripts/preflight/hpsearch/standardized_procedure_stage3.sh ARCH=$ARCH D_TRAIN=$D_TRAIN"
