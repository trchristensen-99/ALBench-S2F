#!/bin/bash
# v22: Feature-aware bias correction strategies.
#
# Goals: address CpG / GC / nucleotide-composition biases beyond what
# cpg_invariance/spectral can do. All strategies stay IN the oracle.
#
# Cells:
#   c290: c91 + GC-balanced neg-aug TSV (GC quintile balanced to Gosai distribution)
#   c291: c91 + CpG-stratified neg-aug (uniform sample across CpG quintiles)
#   c292: c91 + dinuc 3% + GC-stratified weighting via cpg_invariance λ=0.10
#   c293: c91 + mix of intergenic (low-CpG) + Sahu (high-CpG) — covers CpG range
#   c294: c91 + nucleotide-balanced neg-aug (matching ACGT freq to Gosai)
#   c295: c91 + repeat-aware neg-aug (sequences with various repeat density)
#   c296: c91 + intergenic + dinuc + gosai_top10 (3-way: pos+neg-low+neg-mid)
#   c297: c91 + comprehensive 4-way mix: random, dinuc, intergenic, gosai_top

set -euo pipefail

REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"

# Build feature-aware TSVs first
uv run --no-sync python <<PYEOF
import pandas as pd
import numpy as np
from pathlib import Path
REPO = Path("$REPO")

out_dir = REPO / "data/feature_aware_negatives"
out_dir.mkdir(parents=True, exist_ok=True)

def gc(seq):
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)

def cpg(seq):
    s = seq.upper()
    n = max(len(s) - 1, 1)
    return sum(1 for i in range(n) if s[i:i+2] == "CG") / n

# Build CpG-stratified neg-aug: take dinuc-shuffled, bin by CpG, sample uniformly
print("=== Building CpG-stratified neg-aug ===")
neg = pd.read_csv(REPO / "data/synthetic_negatives/dinuc_shuffled_negatives.tsv", sep="\t")
neg["cpg"] = neg["sequence"].apply(cpg)
# 5 quintiles, sample uniform
neg["cpg_bin"] = pd.qcut(neg["cpg"], q=5, labels=False, duplicates="drop")
per_bin = 10000
parts = []
for b in sorted(neg["cpg_bin"].dropna().unique()):
    parts.append(neg[neg["cpg_bin"] == b].sample(min(per_bin, (neg["cpg_bin"] == b).sum()), random_state=42))
df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
df_save = df[["sequence", "K562_log2FC"]].copy()
df_save["category"] = "cpg_stratified_dinuc"
out = out_dir / "cpg_stratified_dinuc.tsv"
df_save.to_csv(out, sep="\t", index=False)
print(f"  Saved {out}: n={len(df_save)}, CpG bins balanced")

# Build GC-balanced (match Gosai full GC distribution)
print("=== Building GC-balanced neg-aug ===")
# Compute Gosai's GC distribution
gosai_path = REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt"
gosai = pd.read_csv(gosai_path, sep="\t")
seq_col = "sequence" if "sequence" in gosai.columns else next(c for c in gosai.columns if "seq" in c.lower())
gosai = gosai.rename(columns={seq_col: "sequence"})
gosai_sub = gosai.sample(5000, random_state=42)
gosai_sub["gc"] = gosai_sub["sequence"].apply(gc)
target_gc_dist = np.histogram(gosai_sub["gc"], bins=10, range=(0, 1))[0]
target_gc_dist = target_gc_dist / target_gc_dist.sum()
# Resample neg to match
neg["gc"] = neg["sequence"].apply(gc)
neg["gc_bin"] = np.floor(neg["gc"] * 10).clip(0, 9).astype(int)
sampled = []
total = 50000
for b, frac in enumerate(target_gc_dist):
    need = int(frac * total)
    avail = neg[neg["gc_bin"] == b]
    if len(avail) > 0:
        sampled.append(avail.sample(min(need, len(avail)), replace=need > len(avail), random_state=42))
df2 = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)
df2_save = df2[["sequence", "K562_log2FC"]].copy()
df2_save["category"] = "gc_balanced_dinuc"
out2 = out_dir / "gc_balanced_dinuc.tsv"
df2_save.to_csv(out2, sep="\t", index=False)
print(f"  Saved {out2}: n={len(df2_save)}")

# Build 3-way mix: low-CpG intergenic + high-CpG Sahu + dinuc
print("=== Building 3-way mix (low-CpG inter + high-CpG Sahu + dinuc) ===")
inter = pd.read_csv(REPO / "data/synthetic_negatives/real_inter_all.tsv", sep="\t")
sahu = pd.read_csv(REPO / "data/synthetic_negatives_sahu/dinuc_shuffled_sahu.tsv", sep="\t")
n_each = 20000
parts3 = [
    inter.sample(min(n_each, len(inter)), random_state=42),
    sahu.sample(min(n_each, len(sahu)), random_state=42),
    neg[["sequence", "K562_log2FC", "category"]].sample(n_each, random_state=42),
]
df3 = pd.concat(parts3).sample(frac=1, random_state=42).reset_index(drop=True)
out3 = out_dir / "mix3way_inter_sahu_dinuc.tsv"
df3.to_csv(out3, sep="\t", index=False)
print(f"  Saved {out3}: n={len(df3)}")

# Build 4-way: pos high + dinuc + intergenic + Sahu (covers ALL sequence space)
print("=== Building 4-way mix (POS + dinuc + intergenic + Sahu) ===")
pos = pd.read_csv(REPO / "data/positive_augmentation/gosai_top10pct.tsv", sep="\t")
n_each = 15000
parts4 = [
    pos.sample(min(n_each, len(pos)), random_state=42)[["sequence", "K562_log2FC", "category"]],
    neg.sample(n_each, random_state=42)[["sequence", "K562_log2FC", "category"]],
    inter.sample(min(n_each, len(inter)), random_state=42),
    sahu.sample(min(n_each, len(sahu)), random_state=42),
]
df4 = pd.concat(parts4).sample(frac=1, random_state=42).reset_index(drop=True)
out4 = out_dir / "mix4way_full_spectrum.tsv"
df4.to_csv(out4, sep="\t", index=False)
print(f"  Saved {out4}: n={len(df4)}, mean K562_log2FC={df4['K562_log2FC'].mean():+.3f}")
PYEOF

OUT=$REPO/outputs/oracle_neg_sweep/debias_sweep_v22
mkdir -p $OUT
S1=$REPO/outputs/oracle_full_856k/s1/oracle_0
FA=$REPO/data/feature_aware_negatives

uv run --no-sync python <<PYEOF
import json
from pathlib import Path

def cfg(label, tsv, frac, mode, lam):
    return {
        "label": label,
        "fold_id": 0, "n_folds": 10,
        "stage1_dir": "$S1",
        "output_dir": f"outputs/oracle_neg_sweep/debias_sweep_v22/{label}/fold_0",
        "epochs": 80, "patience": 15,
        "extra_overrides": [
            f"++negatives_path={tsv}",
            f"++neg_fraction={frac}",
            f"++debias_mode={mode}",
            f"++debias_lambda={lam}",
            "++unfreeze_encoder_blocks=[0,1,2]",
        ],
    }

FA = "$FA"
configs = [
    cfg("c290_gc_balanced",        f"{FA}/gc_balanced_dinuc.tsv",        0.03, "cpg_invariance", 0.05),
    cfg("c291_cpg_stratified",     f"{FA}/cpg_stratified_dinuc.tsv",     0.03, "cpg_invariance", 0.05),
    cfg("c292_cpg_strat_lam10",    f"{FA}/cpg_stratified_dinuc.tsv",     0.03, "cpg_invariance", 0.10),
    cfg("c293_mix3way",            f"{FA}/mix3way_inter_sahu_dinuc.tsv", 0.05, "cpg_invariance", 0.05),
    cfg("c296_mix4way_full",       f"{FA}/mix4way_full_spectrum.tsv",    0.06, "cpg_invariance", 0.05),
    cfg("c297_mix4way_lam10",      f"{FA}/mix4way_full_spectrum.tsv",    0.06, "cpg_invariance", 0.10),
]

CFG = "$OUT/configs.json"
Path(CFG).write_text(json.dumps(configs, indent=2))
print(f"  wrote {len(configs)} v22 configs")
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
    echo "#SBATCH --job-name=pf_debias_v22_b${tag}"
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
    echo "  v22_b${tag}: $JID"
done
