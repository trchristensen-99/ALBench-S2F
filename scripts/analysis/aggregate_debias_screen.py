"""Aggregate 6-variant debias screen results across:
  - test_metrics.json: in_distribution, snv_abs, snv_delta, ood (positive panels, real labels)
  - bias_eval.json: random_dna, shuffled, intergenic (negative panels — μ, σ, pct_positive)

Reports per-variant scoreboard ranked by combined criteria.
"""
import json
from pathlib import Path

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

VARIANTS = [
    "sahu_bin_constraint",
    "moment_match_neg",
    "moment_plus_sahu",
    "adaptive_target_match",
    "adaptive_plus_sahu",
    "adaptive_plus_invariance",
]

# Targets (per Sahu-corrected, ctrl_neg-anchored, CpG-adaptive expected)
TARGETS = {
    "intergenic": (0.27, 0.49),
    "shuffled":   (0.30, 0.50),
    "random_dna": (0.50, 0.50),
}

print(f"{'variant':<30} | {'val_p':<6} | {'id_r':<6} | {'snv_r':<6} | {'ood_r':<6} | {'rand μ':<8} | {'shuf μ':<8} | {'int μ':<8} | dist_loss")
print("-" * 130)

results = []
for variant in VARIANTS:
    tm_path = REPO / f"outputs/debias_screen/{variant}/fold_0/test_metrics.json"
    be_path = REPO / f"outputs/debias_screen/{variant}/fold_0/bias_eval.json"
    if not tm_path.exists() or not be_path.exists():
        print(f"{variant:<30} | (still training)")
        continue
    tm = json.loads(tm_path.read_text())
    be = json.loads(be_path.read_text())
    val_p = tm.get("best_val_pearson", 0)
    id_r = tm["test_metrics"]["in_distribution"]["pearson_r"]
    snv_r = tm["test_metrics"]["snv_abs"]["pearson_r"]
    ood_r = tm["test_metrics"]["ood"]["pearson_r"]
    # Distribution loss: sum of squared (μ-target_μ, σ-target_σ) across 3 negative panels
    dist_loss = 0
    for set_name, (target_mu, target_sigma) in TARGETS.items():
        s = be.get(set_name, {})
        if "mean" not in s: continue
        dist_loss += (s["mean"] - target_mu)**2 + (s["std"] - target_sigma)**2
    rand_mu = be.get("random_dna", {}).get("mean", float("nan"))
    shuf_mu = be.get("shuffled", {}).get("mean", float("nan"))
    int_mu = be.get("intergenic", {}).get("mean", float("nan"))
    print(f"{variant:<30} | {val_p:.4f} | {id_r:.4f} | {snv_r:.4f} | {ood_r:.4f} | {rand_mu:+.3f}   | {shuf_mu:+.3f}   | {int_mu:+.3f}   | {dist_loss:.3f}")
    results.append({"variant": variant, "val_p": val_p, "id_r": id_r, "snv_r": snv_r, "ood_r": ood_r,
                    "rand_mu": rand_mu, "shuf_mu": shuf_mu, "int_mu": int_mu, "dist_loss": dist_loss})

if results:
    print()
    # Rank by combined score: prefer high accuracy + low dist_loss
    # Score = α * (id_r + snv_r + ood_r) - β * dist_loss
    print("Ranked by composite (accuracy − distance loss):")
    for r in sorted(results, key=lambda x: -(x["id_r"] + x["ood_r"] - x["dist_loss"])):
        score = r["id_r"] + r["ood_r"] - r["dist_loss"]
        print(f"  {r['variant']:<30}: score={score:+.3f}  (id_r={r['id_r']:.4f}  ood_r={r['ood_r']:.4f}  dist_loss={r['dist_loss']:.3f})")

# Save
out = REPO / "results/debias_variants/screen_summary.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved {out}")
