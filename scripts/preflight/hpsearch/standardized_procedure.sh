#!/bin/bash
# ============================================================================
# Standardized HP-search procedure for one (ARCH, D_TRAIN) cell.
#
# Stages:
#   1. PARALLEL COARSE SEARCH (3 SLURM jobs in parallel — random + optuna + pbt)
#   2. AUTORESEARCH REFINEMENT (depends on Stage 1; LLM-driven, currently stub)
#   3. AUGMENTATION SWEEP (depends on Stage 2)
#   4. SEED VARIANCE (depends on Stage 3)
#   5. COVERAGE AUDIT (lightweight CPU job, depends on Stage 4)
#
# Modes:
#   SMOKE=1   ↳ Tiny sweep to validate the pipeline plumbing end-to-end. Runs
#               in ~20-30 min on H100 instead of ~3-4 h. Uses:
#                 - 8 trials per strategy (Stage 1) vs 30
#                 - 12 configs (Stage 2) vs 30
#                 - 6 configs (Stage 3) vs 27
#                 - 2 seeds × 1 config (Stage 4) vs 3 seeds × 2 configs
#                 - epochs=25 (early-stop hits ~ep 30 anyway) vs 60
#                 - val_subsample=5000 vs full 57k
#                 - k_parallel=8 + multi-GPU (Stage 1) for max throughput
#   SMOKE=0   ↳ Full procedure. Default.
#
# Transfer mode (independent of SMOKE):
#   ANCHOR_DIR=<...std_…_dXXX>  ↳ Skip Stages 1+2; reuse anchor's winners.
#
# Multi-GPU per stage (independent):
#   N_GPUS=4  ↳ Request N GPUs per Stage 1 job and round-robin trials.
#               Default 1.
#
# Usage examples:
#   # Smoke test (~25 min)
#   SMOKE=1 ARCH=legnet D_TRAIN=20000 bash scripts/preflight/hpsearch/standardized_procedure.sh
#
#   # Full procedure with 4 GPUs per Stage 1 job
#   N_GPUS=4 ARCH=legnet D_TRAIN=20000 bash scripts/preflight/hpsearch/standardized_procedure.sh
#
#   # Transfer mode
#   ANCHOR_DIR=results/preflight/hpsearch/std_legnet_d20000 \
#     ARCH=legnet D_TRAIN=5000 bash scripts/preflight/hpsearch/standardized_procedure.sh
# ============================================================================

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

: "${ARCH:?ARCH required (legnet | dream_attn)}"
: "${D_TRAIN:?D_TRAIN required}"
ANCHOR_DIR=${ANCHOR_DIR:-}
SMOKE=${SMOKE:-0}
N_GPUS=${N_GPUS:-1}

if [ "$SMOKE" = "1" ]; then
    N_STAGE1_TRIALS=${N_STAGE1_TRIALS:-8}
    AUTORESEARCH_CONFIGS=${AUTORESEARCH_CONFIGS:-12}
    AUG_SWEEP_CONFIGS=${AUG_SWEEP_CONFIGS:-6}
    SEED_VAR_REPS=${SEED_VAR_REPS:-2}
    MAX_EPOCHS=${MAX_EPOCHS:-25}
    PATIENCE=${PATIENCE:-6}
    VAL_SUBSAMPLE=${VAL_SUBSAMPLE:-5000}
    K_PARALLEL=${K_PARALLEL:-8}
    STAGE_TIME=${STAGE_TIME:-02:00:00}
    TAG="_smoke"
else
    N_STAGE1_TRIALS=${N_STAGE1_TRIALS:-30}
    AUTORESEARCH_CONFIGS=${AUTORESEARCH_CONFIGS:-30}
    AUG_SWEEP_CONFIGS=${AUG_SWEEP_CONFIGS:-27}
    SEED_VAR_REPS=${SEED_VAR_REPS:-3}
    MAX_EPOCHS=${MAX_EPOCHS:-60}
    PATIENCE=${PATIENCE:-15}
    VAL_SUBSAMPLE=${VAL_SUBSAMPLE:-0}
    K_PARALLEL=${K_PARALLEL:-4}
    STAGE_TIME=${STAGE_TIME:-08:00:00}
    TAG=""
fi

OUT_BASE="results/preflight/hpsearch/std_${ARCH}_d${D_TRAIN}${TAG}"
mkdir -p "$OUT_BASE"
echo "[procedure] $(date) ARCH=$ARCH D_TRAIN=$D_TRAIN SMOKE=$SMOKE N_GPUS=$N_GPUS"
echo "[procedure] OUT_BASE=$OUT_BASE"
echo "[procedure] N_STAGE1=$N_STAGE1_TRIALS MAX_EP=$MAX_EPOCHS K_PAR=$K_PARALLEL VAL_SUB=$VAL_SUBSAMPLE"
echo "[procedure] ANCHOR_DIR=${ANCHOR_DIR:-<none>}"

# Common SLURM wrapper params
CPUS=$(( N_GPUS * 4 + 2 ))
MEM=$(( N_GPUS * 40 ))G

# Helper: submit a Ray Tune strategy (each #SBATCH directive on its own line)
submit_raytune() {
    local strategy=$1
    local jobname="std${TAG}_${ARCH}_d${D_TRAIN}_${strategy}"
    local out="${OUT_BASE}/stage1_${strategy}"
    mkdir -p "$out"
    local jobfile=$(mktemp)
    cat > "$jobfile" <<EOF
#!/bin/bash
#SBATCH --job-name=$jobname
#SBATCH --output=$REPO/logs/%x-%j.out
#SBATCH --error=$REPO/logs/%x-%j.err
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:$N_GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$STAGE_TIME

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache
export EVAL_BATCH_MULT=2

python -m scripts.preflight.hpsearch.raytune_search \\
    --strategy $strategy --arch $ARCH --d_train $D_TRAIN \\
    --n_trials $N_STAGE1_TRIALS --max_epochs $MAX_EPOCHS --patience $PATIENCE \\
    --gpus $N_GPUS --trials_per_gpu $K_PARALLEL \\
    --output_dir $out
EOF
    local jid=$($SBATCH --parsable "$jobfile")
    rm -f "$jobfile"
    echo "$jid"
}

# Helper: submit a config-list runner (for stages 2-4)
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
#SBATCH --partition=gpuq
#SBATCH --qos=default
#SBATCH --gres=gpu:$N_GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$time_lim

cd $REPO
source .venv/bin/activate
export PYTHONPATH="\$PWD"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HP_FAST=1
export HP_CACHE_DIR=\$PWD/outputs/tensor_cache
export EVAL_BATCH_MULT=2
export N_GPUS=$N_GPUS

uv run --no-sync python -u scripts/preflight/parallel_gpu_runner.py "$configs_path" $k_parallel
EOF
    local jid=$($SBATCH --parsable $dep_flag "$jobfile")
    rm -f "$jobfile"
    echo "$jid"
}

source .venv/bin/activate
S1_DEP=""

if [ -z "$ANCHOR_DIR" ]; then
    echo "[stage1] submitting random + optuna + pbt in parallel..."
    S1_RANDOM=$(submit_raytune random)
    S1_OPTUNA=$(submit_raytune optuna)
    S1_PBT=$(submit_raytune pbt)
    echo "  stage1 jobs: random=$S1_RANDOM  optuna=$S1_OPTUNA  pbt=$S1_PBT"
    S1_DEP="afterany:${S1_RANDOM}:${S1_OPTUNA}:${S1_PBT}"
else
    echo "[stage1] SKIPPED (transfer mode from $ANCHOR_DIR)"
fi

# Stage 2 digest job (writes the leaderboard for AutoResearch input)
S2_OUT="${OUT_BASE}/stage2_autoresearch"
mkdir -p "$S2_OUT"
DIGEST_JOBFILE=$(mktemp)
cat > "$DIGEST_JOBFILE" <<EOF
#!/bin/bash
#SBATCH --job-name=std${TAG}_${ARCH}_d${D_TRAIN}_digest
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
echo "[digest] Leaderboard for $ARCH D=$D_TRAIN written. AutoResearch trigger should run next."
EOF
DIGEST_DEP=""
[ -n "$S1_DEP" ] && DIGEST_DEP="--dependency=$S1_DEP"
S2_JID=$($SBATCH --parsable $DIGEST_DEP "$DIGEST_JOBFILE")
rm -f "$DIGEST_JOBFILE"
echo "[stage2] digest job → $S2_JID"

# (Stage 3 + Stage 4 stubs — assembled post-hoc when prior stages land)
echo "[stage3] aug-sweep / [stage4] seed-variance will be assembled after Stage 2 completes."
echo ""
echo "[procedure] Stage 1+2 submitted. To check status:"
echo "  squeue -u christen -h | grep std${TAG}_${ARCH}_d${D_TRAIN}"
echo "[procedure] To monitor leaderboard live:"
echo "  python -m scripts.preflight.hpsearch.aggregate_trials"
