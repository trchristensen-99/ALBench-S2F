"""Quick-dump utility to show test_metrics + bias_eval for a sweep dir.

Reports prediction MEANS and (in parens) RESIDUALS vs the empirical
target derived from real K562 MPRA measurements of those sequence types
in our negative-augmentation TSVs (consistent with Sahu et al. 2022 and
Agarwal et al. — random/intergenic in episomal MPRA average -0.5 to -1.0
log2FC because they lack regulatory motifs).

Empirical targets (from data/synthetic_negatives/*.tsv label columns):
- random_dna   = -0.45  (random_negatives.tsv mean K562_log2FC)
- shuffled     = -0.45  (dinuc_shuffled_negatives.tsv mean)
- intergenic   = -0.75  (real_inter_negative_only.tsv mean)

Residual = mean(prediction) - empirical_target. Closer to 0 = less biased.

Usage:
    uv run --no-sync python scripts/preflight/_dump_sweep_results.py outputs/oracle_neg_sweep/debias_sweep_v4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Empirical means from the actual MPRA measurements in our neg-aug TSVs
TARGETS = {
    "random_dna": -0.45,
    "shuffled": -0.45,
    "intergenic": -0.75,
}


def main():
    base = Path(sys.argv[1])
    rows = []
    for d in sorted(base.glob("c*/fold_0")):
        tm = d / "test_metrics.json"
        if not tm.exists():
            continue
        name = d.parent.name
        m = json.loads(tm.read_text())["test_metrics"]
        be = d / "bias_eval.json"
        b = json.loads(be.read_text()) if be.exists() else {}
        rows.append(
            {
                "name": name,
                "in_dist": m["in_distribution"]["pearson_r"],
                "ood": m["ood"]["pearson_r"],
                "ood_mse": m["ood"]["mse"],
                "rdna": b.get("random_dna", {}).get("mean", 0),
                "shuffled": b.get("shuffled", {}).get("mean", 0),
                "intergenic": b.get("intergenic", {}).get("mean", 0),
            }
        )
    print(
        f"{'name':<35}  {'in_d':>6}  {'OOD':>6}  {'OOD_MSE':>7}  "
        f"{'rdna(resid)':>14}  {'shuf(resid)':>14}  {'inter(resid)':>14}"
    )
    print(
        f"{'(targets)':<35}  {'-':>6}  {'-':>6}  {'-':>7}  "
        f"{TARGETS['random_dna']:>+7.2f}      {TARGETS['shuffled']:>+7.2f}      {TARGETS['intergenic']:>+7.2f}    "
    )
    for r in rows:
        rdna_resid = r["rdna"] - TARGETS["random_dna"]
        shuf_resid = r["shuffled"] - TARGETS["shuffled"]
        inter_resid = r["intergenic"] - TARGETS["intergenic"]
        print(
            f"{r['name']:<35}  {r['in_dist']:>6.3f}  {r['ood']:>6.3f}  {r['ood_mse']:>7.2f}  "
            f"{r['rdna']:>+5.2f}({rdna_resid:>+5.2f})  "
            f"{r['shuffled']:>+5.2f}({shuf_resid:>+5.2f})  "
            f"{r['intergenic']:>+5.2f}({inter_resid:>+5.2f})"
        )


if __name__ == "__main__":
    main()
