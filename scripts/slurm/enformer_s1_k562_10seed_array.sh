#!/bin/bash
# 10-seed Enformer S1 probing ensemble for K562.
# Each task = a different seed; all use chr_split=True (test=chr7+13).
# Cheap: head training on cached embeddings, ~30-60 min/seed.
# Wall via SLURM array = ~30-60 min for all 10.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

OUT_BASE=results/snv_eval/enformer_s1_k562_10seed
mkdir -p "$OUT_BASE" "$REPO/logs"

JOBFILE=$(mktemp)
cat > "$JOBFILE" <<'EOF'
#!/bin/bash
#SBATCH --job-name=enf_s1_k562_seed
#SBATCH --output=__REPO__/logs/%x-%A_%a.out
#SBATCH --error=__REPO__/logs/%x-%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0-9

cd __REPO__
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5 2>/dev/null || true
source .venv/bin/activate
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

SEEDS=(42 43 44 45 46 47 48 49 50 51)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=__OUT_BASE__/seed${SEED}
mkdir -p "$OUT_DIR"

if [ -f "$OUT_DIR/result.json" ]; then
    echo "[skip seed $SEED] result.json already exists"
    exit 0
fi

echo "[enformer S1 seed $SEED] out=$OUT_DIR"

uv run --no-sync python experiments/train_foundation_cached.py \
    ++model_name=enformer \
    ++cache_dir=outputs/chr_split/k562/enformer_cached_v3/embedding_cache \
    ++embed_dim=3072 \
    ++output_dir=$OUT_DIR \
    ++cell_line=k562 \
    ++data_path=data/k562 \
    ++chr_split=True \
    ++include_alt_alleles=True \
    ++seed=$SEED \
    ++lr=0.0001 \
    ++weight_decay=0.0001 \
    ++dropout=0.3 \
    ++hidden_dim=512 \
    ++epochs=100 \
    ++batch_size=512 \
    ++early_stop_patience=10 \
    ++rc_aug=True
EOF

sed -i.bak "s|__REPO__|$REPO|g; s|__OUT_BASE__|$OUT_BASE|g" "$JOBFILE"
rm -f "$JOBFILE.bak"

JID=$($SBATCH --parsable "$JOBFILE")
rm -f "$JOBFILE"
echo "Enformer S1 K562 10-seed array → $JID (10 parallel seeds on slow_nice)"
