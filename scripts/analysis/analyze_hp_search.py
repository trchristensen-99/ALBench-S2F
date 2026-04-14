#!/usr/bin/env python3
"""Analyze exhaustive HP search results.

Reads outputs/exhaustive_hp_search/ and produces:
1. Comparison table: method × (strategy, size) → val_mean, val_std, consistency
2. Ranking: which method is most robust across all conditions
3. Recommendation: single best method for production use

Usage:
    python scripts/analysis/analyze_hp_search.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "outputs" / "exhaustive_hp_search"


def load_all_results():
    """Load all exhaustive HP search results."""
    results = {}
    for f in RESULTS_DIR.rglob("*.json"):
        d = json.loads(f.read_text())
        method = d["method"]
        strategy = d["strategy"]
        n_train = d["n_train"]
        results[(method, strategy, n_train)] = d
    return results


def print_comparison_table(results):
    """Print method comparison table."""
    methods = sorted(set(m for m, _, _ in results))
    conditions = sorted(set((s, n) for _, s, n in results))

    print("\n" + "=" * 100)
    print("HP SEARCH METHOD COMPARISON")
    print("=" * 100)

    # Header
    header = f"{'Method':25s}"
    for strat, n in conditions:
        header += f" | {strat[:6]}/{n // 1000}K"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:25s}"
        for strat, n in conditions:
            key = (method, strat, n)
            if key not in results:
                row += " |    —     "
                continue
            d = results[key]

            if method == "ensemble_3x20":
                ens = d.get("ensemble", {})
                val = ens.get("consensus_mean_val", 0)
                std = ens.get("consensus_std_val", 0)
                row += f" | {val:.3f}±{std:.3f}"
            else:
                c = d.get("consistency", {})
                val = c.get("val_mean", 0)
                std = c.get("val_std", 0)
                row += f" | {val:.3f}±{std:.3f}"
        print(row)


def rank_methods(results):
    """Rank methods by robustness across all conditions."""
    method_scores = defaultdict(list)
    method_consistency = defaultdict(list)

    for (method, strat, n), d in results.items():
        if method == "ensemble_3x20":
            ens = d.get("ensemble", {})
            method_scores[method].append(ens.get("consensus_mean_val", 0))
            method_consistency[method].append(ens.get("consensus_std_val", 1))
        else:
            c = d.get("consistency", {})
            cross = d.get("cross_evaluation", [])

            # Use cross-eval mean as the "true" score
            if cross:
                best_cross = max(cross, key=lambda x: x["cross_eval_mean"])
                method_scores[method].append(best_cross["cross_eval_mean"])
            else:
                method_scores[method].append(c.get("val_mean", 0))

            method_consistency[method].append(c.get("val_std", 1))

    print("\n" + "=" * 80)
    print("METHOD RANKING (by mean cross-eval performance)")
    print("=" * 80)
    print(
        f"{'Rank':>4s}  {'Method':25s}"
        f"  {'Mean Val':>8s}  {'Std Val':>8s}"
        f"  {'Mean Consistency':>15s}  {'N Cond':>6s}"
    )
    print("-" * 80)

    ranked = sorted(
        method_scores.items(),
        key=lambda x: np.mean(x[1]),
        reverse=True,
    )
    for rank, (method, scores) in enumerate(ranked, 1):
        consist = method_consistency[method]
        print(
            f"{rank:4d}  {method:25s}"
            f"  {np.mean(scores):8.4f}  {np.std(scores):8.4f}"
            f"  {np.mean(consist):15.4f}  {len(scores):6d}"
        )

    # Best method
    if ranked:
        best = ranked[0][0]
        print(f"\nRECOMMENDED METHOD: {best}")
        print(f"  Mean performance: {np.mean(ranked[0][1]):.4f}")
        print(f"  Mean consistency (lower=better): {np.mean(method_consistency[best]):.4f}")


def detailed_method_report(results, method):
    """Print detailed report for a specific method."""
    print(f"\n{'=' * 60}")
    print(f"DETAILED REPORT: {method}")
    print(f"{'=' * 60}")

    for (m, strat, n), d in sorted(results.items()):
        if m != method:
            continue
        print(f"\n  {strat} n={n}:")

        if method == "ensemble_3x20":
            ens = d.get("ensemble", {})
            print(f"    Consensus val: {ens.get('consensus_mean_val', 0):.4f}")
            print(f"    HP: {ens.get('consensus_hp', {})}")
        else:
            c = d.get("consistency", {})
            print(f"    Val: {c.get('val_mean', 0):.4f} ± {c.get('val_std', 0):.4f}")
            print(f"    LR ratio: {c.get('lr_ratio', 0):.1f}x")
            print(f"    BS agreement: {c.get('bs_agreement', 0):.0%}")
            print(f"    LRs: {c.get('lr_per_iter', [])}")
            print(f"    BSs: {c.get('bs_per_iter', [])}")

            cross = d.get("cross_evaluation", [])
            if cross:
                print("    Cross-evaluation:")
                for ce in cross:
                    print(
                        f"      lr={ce['hp']['lr']:.5f} bs={ce['hp']['batch_size']}:"
                        f" orig={ce['original_val']:.4f}"
                        f" cross={ce['cross_eval_mean']:.4f}±{ce['cross_eval_std']:.4f}"
                    )


def main():
    if not RESULTS_DIR.exists():
        print(f"No results found in {RESULTS_DIR}")
        return

    results = load_all_results()
    if not results:
        print("No result files found")
        return

    print(f"Loaded {len(results)} results")
    print_comparison_table(results)
    rank_methods(results)

    # Detailed reports for top methods
    method_scores = defaultdict(list)
    for (method, _, _), d in results.items():
        if method == "ensemble_3x20":
            ens = d.get("ensemble", {})
            method_scores[method].append(ens.get("consensus_mean_val", 0))
        else:
            cross = d.get("cross_evaluation", [])
            if cross:
                best_cross = max(cross, key=lambda x: x["cross_eval_mean"])
                method_scores[method].append(best_cross["cross_eval_mean"])

    ranked = sorted(method_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for method, _ in ranked[:3]:
        detailed_method_report(results, method)


if __name__ == "__main__":
    main()
