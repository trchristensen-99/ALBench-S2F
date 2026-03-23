#!/usr/bin/env python3
"""Investigate near-zero OOD performance for AG oracle → DREAM student (dream_rnn_ag).

Hypotheses:
1. Bug: wrong oracle config or label format
2. Real: AG oracle labels are too smooth for DREAM-RNN to learn OOD generalization
3. Data: insufficient training data (only up to n=20k)

Run:
    python scripts/analysis/investigate_dream_ag_ood.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("Investigation: AG oracle → DREAM student (dream_rnn_ag) OOD")
    print("=" * 70)

    for task in ["k562", "yeast"]:
        config_dir = REPO / "outputs" / "exp1_1" / task / "dream_rnn_ag"
        if not config_dir.exists():
            print(f"\n{task}: no dream_rnn_ag directory")
            continue

        print(f"\n{'=' * 50}")
        print(f"Task: {task.upper()} — dream_rnn_ag")
        print(f"{'=' * 50}")

        # Collect all results
        results = []
        for rj in config_dir.rglob("result.json"):
            try:
                r = json.loads(rj.read_text())
                r["_path"] = str(rj.relative_to(config_dir))
                results.append(r)
            except Exception:
                pass

        print(f"Total results: {len(results)}")

        # Group by reservoir and n_train
        by_res_n = defaultdict(list)
        for r in results:
            by_res_n[(r["reservoir"], r["n_train"])].append(r)

        print(f"\nReservoir × n_train combinations: {len(by_res_n)}")

        # Print all results with test metrics
        print(f"\n{'Reservoir':>25s} {'N':>7s} {'ValR':>6s} {'OOD':>7s} {'ID':>7s} {'SNV_abs':>7s}")
        print("-" * 65)

        for res, n in sorted(by_res_n.keys(), key=lambda x: (x[0], x[1])):
            for r in by_res_n[(res, n)]:
                val_r = r.get("val_pearson_r", 0)
                ood = r["test_metrics"].get("ood", {}).get("pearson_r", None)
                in_dist = r["test_metrics"].get("in_dist", {}).get("pearson_r", None)
                snv = r["test_metrics"].get("snv_abs", {}).get("pearson_r", None)

                ood_s = f"{ood:.4f}" if ood is not None else "--"
                id_s = f"{in_dist:.4f}" if in_dist is not None else "--"
                snv_s = f"{snv:.4f}" if snv is not None else "--"

                print(f"{res:>25s} {n:>7,d} {val_r:>6.4f} {ood_s:>7s} {id_s:>7s} {snv_s:>7s}")

        # Check label statistics from cached NPZ files
        print(f"\n--- Label statistics from cached oracle labels ---")
        for npz_path in sorted(config_dir.rglob("oracle_labels.npz")):
            rel = npz_path.relative_to(config_dir)
            data = np.load(npz_path, allow_pickle=True)
            labels = data["labels"]
            seqs = list(data["sequences"])
            print(
                f"  {str(rel):50s}  "
                f"n={len(labels):>6,d}  "
                f"mean={np.mean(labels):.3f}  std={np.std(labels):.3f}  "
                f"range=[{np.min(labels):.2f}, {np.max(labels):.2f}]  "
                f"seq_len={len(seqs[0]) if seqs else '?'}"
            )

        # Compare with DREAM→DREAM for same task
        dream_dream = REPO / "outputs" / "exp1_1" / task / "dream_rnn_dream_rnn"
        if dream_dream.exists():
            print(f"\n--- Comparison: dream_rnn_dream_rnn (same student, different oracle) ---")
            dd_results = []
            for rj in dream_dream.rglob("result.json"):
                try:
                    dd_results.append(json.loads(rj.read_text()))
                except Exception:
                    pass

            by_n = defaultdict(list)
            for r in dd_results:
                by_n[r["n_train"]].append(r)

            for n in sorted(by_n.keys()):
                ood_vals = [
                    r["test_metrics"].get("ood", {}).get("pearson_r")
                    for r in by_n[n]
                    if r["test_metrics"].get("ood", {}).get("pearson_r") is not None
                ]
                if ood_vals:
                    print(
                        f"  n={n:>7,d}  "
                        f"OOD: {np.mean(ood_vals):.3f}±{np.std(ood_vals):.3f}  "
                        f"({len(ood_vals)} runs)"
                    )

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
