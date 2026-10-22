"""Map expected activity distributions for ALL 6 eval panels:

  POSITIVE (real Gosai labels — distribution should be preserved):
    1. genomic (chr 7+13 ref) — 32k
    2. SNV-delta (mono-allelic) — 30k
    3. OOD designed — 22k

  NEGATIVE CONTROLS (target for debiased oracle):
    4. Gosai ctrl_neg (real labels) — 503
    5. dinuc_shuffled (no labels, expected match to ctrl_neg distribution) — 5000
    6. random_uniform (no labels, expected match + Sahu bin ordering) — 5000
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")


def summary(name, vals, extra=None):
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    out = {
        "name": name,
        "n": int(len(v)),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "q05": float(np.quantile(v, 0.05)),
        "q25": float(np.quantile(v, 0.25)),
        "q50": float(np.quantile(v, 0.5)),
        "q75": float(np.quantile(v, 0.75)),
        "q95": float(np.quantile(v, 0.95)),
        "min": float(v.min()),
        "max": float(v.max()),
    }
    if extra: out.update(extra)
    return out


panels = {}

# 1. Genomic chr 7+13 ref
chr_oracle = np.load(REPO / "outputs/oracle_pseudolabels_k562_chr_split_ag_s2/chr_7_13_all_oracle.npz")
allele = chr_oracle["allele"]
true_label = chr_oracle["true_label"]
ref_mask = (allele == "R")
ref_vals = true_label[ref_mask]
panels["genomic_ref_32k"] = summary("genomic_ref_32k (chr7+13, real K562 log2FC)", ref_vals)

# 2. SNV-delta (mono-allelic)
snv = np.load(REPO / "outputs/oracle_pseudolabels_k562_chr_split_ag_s2/snv_pairs_full_oracle.npz")
delta_oracle = snv["delta_oracle_mean"]  # NOTE: these are oracle deltas; need to find real labels
mono_mask = snv["mono_mask"]
# Read real snv labels
snv_csv = pd.read_csv(REPO / "data/k562/test_sets/test_snv_pairs.tsv", sep="\t")
real_delta = snv_csv["K562_log2FC_alt"] - snv_csv["K562_log2FC_ref"] if "K562_log2FC_alt" in snv_csv.columns else None
if real_delta is None:
    # fallback to oracle delta
    real_delta = delta_oracle[mono_mask]
    panels["snv_delta_30k_mono"] = summary("snv_delta_30k_mono (ORACLE, no real label found)", real_delta)
else:
    panels["snv_delta_30k_mono"] = summary("snv_delta_30k_mono (real K562 delta)", real_delta[mono_mask])

# 3. OOD designed
ood = np.load(REPO / "outputs/oracle_pseudolabels_k562_chr_split_ag_s2/ood_designed_oracle.npz")
ood_real = ood["true_label"]
panels["ood_designed_22k"] = summary("ood_designed_22k (real K562 log2FC)", ood_real)

# 4. Gosai ctrl_neg (real)
ctrl = pd.read_parquet(REPO / "data/k562/gosai_ctrl_neg.parquet")
panels["gosai_ctrl_neg_503"] = summary("gosai_ctrl_neg_503 (real K562 log2FC, genomic intergenic)", ctrl["K562_log2FC"])

# 5. dinuc_shuffled — no real labels; expected to match ctrl_neg distribution
# We compute expected dist by sampling from ctrl_neg with replacement (approximation)
rng = np.random.default_rng(42)
dinuc_target = rng.choice(ctrl["K562_log2FC"].values, size=5000, replace=True)
panels["dinuc_shuffled_5k_TARGET"] = summary(
    "dinuc_shuffled_5k_TARGET (EXPECTED distribution = ctrl_neg empirical, bootstrapped to 5k)",
    dinuc_target,
    extra={"reasoning": "Dinuc-shuffled sequences derived from real CRE-positive sequences but with motifs destroyed should approximate the genomic non-regulatory baseline (Gosai ctrl_neg empirical distribution)."}
)

# 6. random_uniform — no real labels; expected to roughly match ctrl_neg but with Sahu CpG bin ordering
random_target = rng.choice(ctrl["K562_log2FC"].values, size=5000, replace=True)
panels["random_uniform_5k_TARGET"] = summary(
    "random_uniform_5k_TARGET (EXPECTED distribution = ctrl_neg empirical, CpG bin ordering per Sahu)",
    random_target,
    extra={"reasoning": "Truly random sequences should have ctrl_neg-like distribution overall; Sahu N150 STARR-seq shows ~25% activity increase from low to high CpG bins → use bin ordering as soft constraint."}
)

# Print all
print("=" * 80)
print("EXPECTED ACTIVITY DISTRIBUTIONS — 6 EVAL PANELS")
print("=" * 80)
for k, p in panels.items():
    print(f"\n=== {p['name']} ===")
    print(f"  n={p['n']:>7}  mean={p['mean']:+.3f}  std={p['std']:.3f}")
    print(f"  quantiles: q05={p['q05']:+.3f}  q25={p['q25']:+.3f}  q50={p['q50']:+.3f}  q75={p['q75']:+.3f}  q95={p['q95']:+.3f}")
    print(f"  range: [{p['min']:+.3f}, {p['max']:+.3f}]")
    if 'reasoning' in p:
        print(f"  RATIONALE: {p['reasoning']}")

# Save
out_dir = REPO / "outputs/eval_panel_distributions"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "six_panel_expected_distributions.json").write_text(json.dumps(panels, indent=2))
print(f"\nSaved {out_dir / 'six_panel_expected_distributions.json'}")
