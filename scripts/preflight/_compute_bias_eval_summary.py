"""Summarize bias_eval.json statistics across 10-fold oracles + compare to expected."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")

oracles = {
    "baseline": "outputs/stage2_k562_oracle",
    "c28_10fold": "outputs/oracle_neg_sweep/debias_oracle_c28_10fold",
    "c63_10fold": "outputs/oracle_neg_sweep/debias_c63_10fold",
    "c86_10fold": "outputs/oracle_neg_sweep/debias_c86_10fold",
    "c91_10fold": "outputs/oracle_neg_sweep/debias_c91_10fold",
}

# Reference: target distribution mean for each test set
TARGETS = {
    "random_dna": {"mean": 0.0, "note": "uniform random DNA, expected ~0 (no signal)"},
    "shuffled": {"mean": 0.27, "note": "dinuc-shuffled (analog of Gosai ctrl_neg): +0.27 ± 0.49"},
    "intergenic": {"mean": 0.0, "note": "real K562 intergenic, expected ~0 (basal)"},
}

print(f"{'oracle':<14}  {'random_mean':>20}  {'shuffled_mean':>20}  {'intergenic_mean':>20}")
print(f"{'':<14}  {'(target 0.00)':>20}  {'(target +0.27)':>20}  {'(target 0.00)':>20}")
print("-" * 80)

results = {}
for name, p in oracles.items():
    rows = []
    for fold_dir in sorted((REPO / p).glob("fold_*")):
        be = fold_dir / "bias_eval.json"
        if be.exists():
            rows.append(json.loads(be.read_text()))
    if not rows:
        continue

    def stat(key, sub):
        vals = [r.get(key, {}).get(sub, np.nan) for r in rows]
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    metrics = {}
    for cat in ["random_dna", "shuffled", "intergenic"]:
        m, sm = stat(cat, "mean")
        s, _ = stat(cat, "std")
        p_pos, _ = stat(cat, "pct_positive")
        metrics[cat] = {"mean": m, "mean_std": sm, "std": s, "pct_positive": p_pos}

    results[name] = metrics

    # Print row
    rd = metrics["random_dna"]
    sh = metrics["shuffled"]
    ig = metrics["intergenic"]
    print(
        f"{name:<14}  "
        f"{rd['mean']:>+6.2f}±{rd['mean_std']:>4.2f}/std{rd['std']:>4.2f}/{rd['pct_positive']:>2.0f}%+ "
        f"{sh['mean']:>+6.2f}±{sh['mean_std']:>4.2f}/std{sh['std']:>4.2f}/{sh['pct_positive']:>2.0f}%+ "
        f"{ig['mean']:>+6.2f}±{ig['mean_std']:>4.2f}/std{ig['std']:>4.2f}/{ig['pct_positive']:>2.0f}%+"
    )

# Compute "bias residual" — distance from target
print("\n=== Distance from expected (lower = better calibrated) ===")
print(f"{'oracle':<14}  {'|random-0|':>14}  {'|shuf-0.27|':>14}  {'|interg-0|':>14}  {'composite':>10}")
print("-" * 70)
for name, m in results.items():
    rd_err = abs(m["random_dna"]["mean"] - 0.0)
    sh_err = abs(m["shuffled"]["mean"] - 0.27)
    ig_err = abs(m["intergenic"]["mean"] - 0.0)
    composite = (rd_err + sh_err + ig_err) / 3
    print(f"{name:<14}  {rd_err:>14.3f}  {sh_err:>14.3f}  {ig_err:>14.3f}  {composite:>10.3f}")

# Save
(REPO / "results/preflight/bias_eval_summary.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved {REPO}/results/preflight/bias_eval_summary.json")
