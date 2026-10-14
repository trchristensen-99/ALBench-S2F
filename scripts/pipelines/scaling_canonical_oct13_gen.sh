#!/bin/bash
# Generator for the canonical scaling-curve HP-search (Oct 13 2026).
# Emits one sbatch per (reservoir, D) cell + a smoke sbatch + a watchdog,
# all under outputs/scaling_canonical_oct13/. Methodology = llm2x30 BROAD+arch.
# Driver is the CONSOLIDATED experiments/scaling_hp_search.py (single-seed default).
set -euo pipefail

R=/grid/wsbs/home_norepl/christen/ALBench-S2F
BASE=$R/outputs/scaling_canonical_oct13
CACHE=$R/outputs/grid_pools/k562_cumulative
mkdir -p "$BASE/logs"

RES_LIST="genomic zoonomia_flank_primates zoonomia_flank_mammals random gc_matched dinuc_shuffle motif_jaspar motif_ct_enriched motif_grammar evoaug_paper2023 evoaug_prm010 d3_steps64 evo2"
D_LIST="10000 30000 100000"

# QOS round-robin across cells; walltime matched to the qos cap.
QOS_CYCLE=(fast default slow_nice)
declare -A WALL=( [fast]=4:00:00 [default]=12:00:00 [slow_nice]=2-00:00:00 )

# Shared env block (the llm2x30 BROAD + arch methodology lockdown).
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

i=0
J30=0   # parent-shell counter for 30k H100/V100 alternation (NOT a $() subshell)
for D in $D_LIST; do
  for RES in $RES_LIST; do
    CACHE_NPZ=$CACHE/${RES}_d${D}.npz
    if [ ! -f "$CACHE_NPZ" ]; then echo "MISSING cache: $CACHE_NPZ" >&2; exit 2; fi
    QOS=${QOS_CYCLE[$(( i % 3 ))]}
    W=${WALL[$QOS]}
    OUT=$BASE/${RES}_d${D}
    SB=$BASE/sc_${RES}_${D}.sbatch
    # Per-D GPU routing: 10k->V100, 100k->H100, 30k alternates H100/V100.
    case "$D" in
      10000)  GRES="gpu:v100:1" ;;
      100000) GRES="gpu:h100:1" ;;
      30000)  if (( J30 % 2 == 0 )); then GRES="gpu:h100:1"; else GRES="gpu:v100:1"; fi; J30=$((J30+1)) ;;
      *)      GRES="gpu:h100:1" ;;
    esac
    {
      echo '#!/bin/bash'
      echo "#SBATCH --job-name=sc_${RES}_${D}"
      echo "#SBATCH --partition=gpuq --qos=${QOS} --gres=${GRES}"
      echo "#SBATCH --cpus-per-task=8 --mem=90G --time=${W} --requeue"
      echo "#SBATCH --output=$BASE/logs/sc_${RES}_${D}_%j.log"
      emit_env "$OUT" "$RES" "$D"
      echo "uv run --no-sync python experiments/scaling_hp_search.py \\"
      echo "  --strategies llm_autoresearch --rounds 20 --per_strategy_per_round 2 \\"
      echo "  --hp_seed \$ALBENCH_RUN_SEED --D ${D} --reservoir_cache ${CACHE_NPZ} \\"
      echo "  --out_dir ${OUT} --epochs 100 --chr_val"
    } > "$SB"
    i=$((i+1))
    echo "wrote $SB [qos=$QOS wall=$W]"
  done
done

# ---- smoke sbatch: genomic d10000, few rounds, fast, isolated out_dir ----
SMK=$BASE/sc_smoke_genomic_10000.sbatch
SMK_OUT=$BASE/_smoke_genomic_d10000
{
  echo '#!/bin/bash'
  echo "#SBATCH --job-name=sc_smoke"
  echo "#SBATCH --partition=gpuq --qos=fast --gres=gpu:h100:1"
  echo "#SBATCH --cpus-per-task=8 --mem=90G --time=4:00:00 --requeue"
  echo "#SBATCH --output=$BASE/logs/sc_smoke_%j.log"
  emit_env "$SMK_OUT" "genomic" "10000"
  echo "uv run --no-sync python experiments/scaling_hp_search.py \\"
  echo "  --strategies llm_autoresearch --rounds 3 --per_strategy_per_round 2 \\"
  echo "  --hp_seed \$ALBENCH_RUN_SEED --D 10000 --reservoir_cache $CACHE/genomic_d10000.npz \\"
  echo "  --out_dir ${SMK_OUT} --epochs 100 --chr_val"
} > "$SMK"
echo "wrote $SMK (smoke)"

echo "DONE: $i cells + 1 smoke under $BASE"
