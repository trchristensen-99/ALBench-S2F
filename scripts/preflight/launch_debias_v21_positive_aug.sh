#!/bin/bash
# v21: POSITIVE augmentation strategies + feature-aware corrections.
#
# Motivation: classical regression-to-mean causes oracles to under-predict
# high-activity sequences. Adding high-activity Gosai sequences as training
# augmentation (with their REAL labels) should counter this compression.
# All debiasing stays IN the oracle (no post-hoc).
#
# Cells:
#   c280: c91 + gosai_top10pct as POS-aug (uses same neg-aug mechanism but with high labels)
#   c281: c91 + gosai_top25pct as POS-aug
#   c282: c91 + gosai_top5pct as POS-aug (extreme tail)
#   c283: c91 + balanced: dinuc 3% (neg) AND gosai_top25pct 3% (pos)  — needs combined TSV
#   c284: c91 + GC-balanced neg-aug (matched to gosai GC distribution)
#   c285: c91 base + real_inter + Gosai-top25 — handles BOTH ends
#   c286: c91 + Sahu (CpG-calibrated neg) + gosai_top10pct (high-activity)
#   c287: Pure c91 + gosai_top10pct (compare with c280; no other changes)

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# Ensure positive augmentation TSVs exist
if [ ! -f "$REPO/data/positive_augmentation/gosai_top10pct.tsv" ]; then
    echo "=== Building positive-aug TSVs ==="
    uv run --no-sync python scripts/preflight/_make_positive_aug_tsv.py
fi

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v21
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
POS_DIR=$REPO/data/positive_augmentation

# Build combined POS+NEG TSV for c283
uv run --no-sync python <<PYEOF
import pandas as pd
from pathlib import Path
pos = pd.read_csv("$POS_DIR/gosai_top25pct.tsv", sep="\t")
neg = pd.read_csv("$REPO/data/synthetic_negatives/dinuc_shuffled_negatives.tsv", sep="\t")
# Resample to balance — take same N from each
n = min(50000, len(pos), len(neg))
combined = pd.concat([pos.sample(n, random_state=42), neg.sample(n, random_state=42)])
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
out_dir = Path("$REPO/data/positive_augmentation")
out = out_dir / "balanced_pos25_neg_dinuc.tsv"
combined.to_csv(out, sep="\t", index=False)
print(f"  Saved {out}: n={len(combined)}, mean K562_log2FC={combined['K562_log2FC'].mean():+.3f}")

# Combined real_inter + gosai_top25 (intergenic neg + high-activity pos)
inter = pd.read_csv("$REPO/data/synthetic_negatives/real_inter_all.tsv", sep="\t")
combined2 = pd.concat([pos.sample(n, random_state=42), inter.sample(min(n, len(inter)), random_state=42)])
combined2 = combined2.sample(frac=1, random_state=42).reset_index(drop=True)
out2 = out_dir / "balanced_pos25_neg_inter.tsv"
combined2.to_csv(out2, sep="\t", index=False)
print(f"  Saved {out2}: n={len(combined2)}, mean K562_log2FC={combined2['K562_log2FC'].mean():+.3f}")
PYEOF

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, tsv, frac, mode, lam, blocks):
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v21/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path={tsv}",
            f"++neg_fraction={frac}",
            f"++debias_mode={mode}",
            f"++debias_lambda={lam}",
            f"++unfreeze_encoder_blocks={blocks}",
        ],
    }

POS = "$POS_DIR"
configs = [
    cfg("c280_pos10_cpginv",         f"{POS}/gosai_top10pct.tsv",                  0.03, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c281_pos25_cpginv",         f"{POS}/gosai_top25pct.tsv",                  0.03, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c282_pos5_cpginv",          f"{POS}/gosai_top5pct.tsv",                   0.03, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c283_balanced_pos25_dinuc", f"{POS}/balanced_pos25_neg_dinuc.tsv",        0.06, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c285_balanced_pos25_inter", f"{POS}/balanced_pos25_neg_inter.tsv",        0.06, "cpg_invariance", 0.05, "[0,1,2]"),
    cfg("c287_pos10_no_cpginv",      f"{POS}/gosai_top10pct.tsv",                  0.03, "none",            0.0,  "[0,1,2]"),
]

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v21 configs (positive-aug)")
PYEOF

uv run --no-sync python <<PYEOF
import json
from pathlib import Path
configs = json.loads(Path("$OUT/configs.json").read_text())
n = len(configs)
b = (n + 1) // 2
batches = [configs[i:i+b] for i in range(0, n, b)]
for i, batch in enumerate(batches):
    Path(f"$OUT/configs_b{i}.json").write_text(json.dumps(batch, indent=2))
print(f"  split: {[len(b) for b in batches]}")
PYEOF

for tag in 0 1; do
    SCRIPT=$(mktemp)
    {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=pf_debias_v21_b${tag}"
    echo "#SBATCH --output=$REPO/logs/%x-%j.out"
    echo "#SBATCH --error=$REPO/logs/%x-%j.err"
    echo "#SBATCH --partition=gpuq --qos=slow_nice --gres=gpu:h100:1 --cpus-per-task=14 --time=06:00:00 --mem=200G"
    echo "set -euo pipefail; set +u; source /etc/profile.d/modules.sh; set -u"
    echo "module load EB5; cd $REPO"
    echo "export PYTHONPATH=\"\$PWD\""
    echo "source scripts/slurm/setup_hpc_deps.sh"
    echo "uv run --no-sync python scripts/preflight/parallel_ag_s2_runner.py $OUT/configs_b${tag}.json 3"
    } > $SCRIPT
    JID=$(/cm/shared/apps/slurm/current/bin/sbatch --parsable $SCRIPT 2>&1)
    rm -f $SCRIPT
    echo "  v21_b${tag}: $JID"
done
