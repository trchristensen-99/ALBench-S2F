#!/bin/bash
# Add the Evo2 generative-LM reservoir to the canonical scaling grid (Oct 2026).
#
# Emits:
#   - outputs/sc_pools/lbl_evo2.sbatch         GPU oracle_v3 labelling of the Evo2 pool
#                                              into nested 10k/30k/100k caches
#   - outputs/scaling_canonical_oct13/sc_evo2_{10000,30000,100000}.sbatch
#                                              scaling HP-search cells (llm2x30 BROAD+arch)
#
# These reuse the exact env block + driver invocation of the existing canonical cells
# (scripts/pipelines/scaling_canonical_oct13_gen.sh) so the sc_watchdog rotates the new
# cells in alongside the 42 existing ones with no further wiring. QOS is spread one per
# D (fast/default/slow_nice), matching the main generator's round-robin convention.
set -euo pipefail

R=/grid/wsbs/home_norepl/christen/ALBench-S2F
BASE=$R/outputs/scaling_canonical_oct13
CACHE=$R/outputs/grid_pools/k562_cumulative
POOLDIR=$R/outputs/sc_pools
EVO2_POOL=/grid/koo/home/mundewadi/evo2_k562_uncond/samples/evo2_cond_regen_pool_1M.npz
mkdir -p "$BASE/logs" "$POOLDIR/logs"

# ---- labelling sbatch: nested oracle_v3 caches from Yash's Evo2 pool -------------
LBL=$POOLDIR/lbl_evo2.sbatch
cat > "$LBL" <<EOF
#!/bin/bash
#SBATCH --job-name=lbl_evo2
#SBATCH --partition=gpuq --qos=fast --gres=gpu:h100:1
#SBATCH --cpus-per-task=8 --mem=90G --time=04:00:00 --requeue
#SBATCH --output=$POOLDIR/logs/lbl_evo2_%j.log
cd $R
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
source .venv/bin/activate
export PYTHONPATH="\$PWD:\$PWD/experiments" PYTHONUNBUFFERED=1 TQDM_DISABLE=1
# Guard: the oracle_v3 default must reproduce an existing cache's labels before we
# trust a new one (measure, don't assume).
uv run --no-sync python -I scripts/verify_oracle_v3_repro.py \\
  $CACHE/genomic_d10000.npz 2000
# Build the nested 10k/30k/100k Evo2 caches with oracle_v3.
uv run --no-sync python scripts/label_d3_pool.py $EVO2_POOL evo2 0
EOF
echo "wrote $LBL"

# ---- shared env block (identical to scaling_canonical_oct13_gen.sh emit_env) ----
emit_env() {
  local out=$1 res=$2 D=$3
  cat <<EOF
cd $R
set +u; source /etc/profile.d/modules.sh; [ -f ~/.bash_profile ] && source ~/.bash_profile; set -u
source .venv/bin/activate
export PYTHONPATH="\$PWD:\$PWD/experiments" PYTHONUNBUFFERED=1 TQDM_DISABLE=1
export ALBENCH_OUT_DIR=$out
export LLM_FAIL_HARD=1
export ALBENCH_RESERVOIR="reservoir=$res (oracle_v3-labelled, chromosome-split pool, D=$D)"
export LLM_USE_BRIDGE=1 LLM_BRIDGE_MODEL=sonnet LLM_BRIDGE_TIMEOUT=5400
export LLM_PROMPT_STYLE=context_rich LLM_CONTEXT=full LLM_USE_KB=1 LLM_ALLOW_NOVEL_AXES=1 LLM_ALLOW_ARCH_CODE=1
export HP_WB_MENU=32,64,96,128,192,256,384,512,768,1024 HP_KS_MENU=7,9,11,15,19,25
export HP_WD_MIN=1e-8 HP_WD_MAX=1e-1 HP_LR_MIN=1e-5 HP_LR_MAX=1e-2 HP_MEM_BUDGET_GB=70
mkdir -p $out
ALBENCH_RUN_SEED=\$(( (RANDOM<<15 | RANDOM) ))
echo "\$ALBENCH_RUN_SEED" >> $out/hp_seed_history.txt
echo "[seed] hp_seed=\$ALBENCH_RUN_SEED (drawn at submit)"
EOF
}

# ---- scaling cells: one QOS per D (round-robin spread) ---------------------------
RES=evo2
declare -A QOS=( [10000]=fast [30000]=default [100000]=slow_nice )
declare -A WALL=( [fast]=4:00:00 [default]=12:00:00 [slow_nice]=2-00:00:00 )

for D in 10000 30000 100000; do
  CACHE_NPZ=$CACHE/${RES}_d${D}.npz
  Q=${QOS[$D]}
  W=${WALL[$Q]}
  OUT=$BASE/${RES}_d${D}
  SB=$BASE/sc_${RES}_${D}.sbatch
  {
    echo '#!/bin/bash'
    echo "#SBATCH --job-name=sc_${RES}_${D}"
    echo "#SBATCH --partition=gpuq --qos=${Q} --gres=gpu:h100:1"
    echo "#SBATCH --cpus-per-task=8 --mem=90G --time=${W} --requeue"
    echo "#SBATCH --output=$BASE/logs/sc_${RES}_${D}_%j.log"
    emit_env "$OUT" "$RES" "$D"
    echo "uv run --no-sync python experiments/scaling_hp_search.py \\"
    echo "  --strategies llm_autoresearch --rounds 30 --per_strategy_per_round 2 \\"
    echo "  --hp_seed \$ALBENCH_RUN_SEED --D ${D} --reservoir_cache ${CACHE_NPZ} \\"
    echo "  --out_dir ${OUT} --epochs 100 --chr_val"
  } > "$SB"
  echo "wrote $SB [qos=$Q wall=$W]"
done

echo "DONE: evo2 labelling sbatch + 3 scaling cells"
