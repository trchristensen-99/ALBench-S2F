#!/usr/bin/env python
"""Generate a detailed comparison between Malinois (boda2 tutorial) and AlphaGenome results.

Reads:
  - Malinois: outputs/malinois_eval_boda2_tutorial/result.json (chrom_test + hashfrag)
  - AlphaGenome val: scripts/analysis/alphagenome_baseline_val_pearson.json
  - AlphaGenome HashFrag (optional): outputs/ag_hashfrag_results.json
    (create with: uv run python scripts/analysis/export_ag_hashfrag_results.py)

Usage (from repo root, after Malinois eval has finished):
  uv run python scripts/analysis/compare_malinois_alphagenome_results.py
  uv run python scripts/analysis/compare_malinois_alphagenome_results.py --output report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--malinois_json",
        default="outputs/malinois_eval_boda2_tutorial/result.json",
        help="Path to Malinois boda2-tutorial eval result JSON.",
    )
    p.add_argument(
        "--ag_val_json",
        default="scripts/analysis/alphagenome_baseline_val_pearson.json",
        help="Path to AlphaGenome validation Pearson baseline JSON.",
    )
    p.add_argument(
        "--ag_hashfrag_json",
        default=None,
        help="Path to AlphaGenome HashFrag test results JSON (optional). "
        "Default: outputs/ag_hashfrag_results.json if present.",
    )
    p.add_argument(
        "--ag_chrom_test_json",
        default=None,
        help="Path to AlphaGenome chr 7, 13 test results (optional). "
        "Default: outputs/ag_chrom_test_results.json. Generate with: uv run python scripts/analysis/eval_ag_chrom_test.py",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write report to this path (default: print to stdout).",
    )
    p.add_argument(
        "--allow_missing_malinois",
        action="store_true",
        help="If Malinois result JSON is missing, still write report using expected test Pearson 0.88-0.89.",
    )
    return p.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    args = get_args()
    malinois_path = Path(args.malinois_json)
    ag_val_path = Path(args.ag_val_json)
    ag_hashfrag_path = (
        Path(args.ag_hashfrag_json)
        if args.ag_hashfrag_json
        else Path("outputs/ag_hashfrag_results.json")
    )
    ag_chrom_path = (
        Path(args.ag_chrom_test_json)
        if args.ag_chrom_test_json
        else Path("outputs/ag_chrom_test_results.json")
    )

    malinois = load_json(malinois_path)
    if malinois is None and not getattr(args, "allow_missing_malinois", False):
        raise SystemExit(
            f"Malinois results not found at {malinois_path}. Run the boda2-tutorial eval first, or use --allow_missing_malinois."
        )
    if malinois is None:
        malinois = {
            "chrom_test": {"pearson_r": 0.885, "spearman_r": None, "mse": None, "n": 62000},
            "_expected_range": "0.88-0.89",
        }

    ag_val = load_json(ag_val_path)
    if ag_val is None:
        raise SystemExit(f"AlphaGenome val baseline not found at {ag_val_path}.")

    ag_hashfrag = load_json(ag_hashfrag_path)
    ag_chrom = load_json(ag_chrom_path)

    lines = []
    lines.append("# Malinois vs AlphaGenome: K562 evaluation comparison")
    lines.append("")
    lines.append("## 1. Data sources")
    lines.append("")
    lines.append(
        f"- **Malinois (boda2 tutorial):** `{malinois_path}` — original K562 test set (chromosome split chr 7, 13 from DATA-Table_S2)."
    )
    lines.append(
        f"- **AlphaGenome val:** `{ag_val_path}` — validation Pearson R (chr 19, 21, X), no_shift Boda heads."
    )
    if ag_chrom:
        lines.append(
            f"- **AlphaGenome chrom test:** `{ag_chrom_path}` — test Pearson/Spearman/MSE (chr 7, 13), same split as Malinois."
        )
    if ag_hashfrag:
        lines.append(
            f"- **AlphaGenome HashFrag:** `{ag_hashfrag_path}` — optional; not used for primary comparison."
        )
    lines.append("")

    # Original K562 test: chromosome split chr 7, 13 (from DATA-Table_S2), not HashFrag TSVs
    lines.append("## 2. Original K562 test set (chr 7, 13)")
    lines.append("")
    lines.append(
        "Test split from the main K562 dataset (DATA-Table_S2), chromosome-based holdout. Not HashFrag."
    )
    lines.append("")
    chrom = malinois.get("chrom_test")
    if chrom or ag_chrom:
        lines.append("| Model | Pearson R | Spearman R | MSE | n |")
        lines.append("|-------|-----------|------------|-----|---|")
        if chrom:
            sr = chrom.get("spearman_r")
            mse = chrom.get("mse")
            n = chrom.get("n")
            lines.append(
                f"| Malinois (boda2 tutorial) | {chrom['pearson_r']:.4f} | "
                f"{f'{sr:.4f}' if sr is not None else '—'} | "
                f"{f'{mse:.4f}' if mse is not None else '—'} | {n or '—'} |"
            )
        if ag_chrom:
            for head_label, m in sorted(ag_chrom.items()):
                pr = m.get("pearson_r")
                spr = m.get("spearman_r")
                mse = m.get("mse")
                n = m.get("n")
                if pr is not None:
                    lines.append(
                        f"| AlphaGenome ({head_label}) | {pr:.4f} | "
                        f"{f'{spr:.4f}' if spr is not None else '—'} | "
                        f"{f'{mse:.4f}' if mse is not None else '—'} | {n or '—'} |"
                    )
        if malinois and malinois.get("_expected_range"):
            lines.append("")
            lines.append(
                f"*(Malinois test Pearson expected range: {malinois['_expected_range']}; actual run pending artifact on HPC.)*"
            )
        if not chrom and ag_chrom:
            lines.append("")
            lines.append("*(Malinois chrom-test: run eval_malinois_boda2_tutorial to add.)*")
    else:
        lines.append(
            "No chrom-test results. Run eval_ag_chrom_test.py for AlphaGenome; run Malinois eval for baseline."
        )
    lines.append("")

    # Validation: AlphaGenome val Pearson (vs Malinois val from notebook: 0.9131)
    lines.append("## 3. Validation split (chr 19, 21, X)")
    lines.append("")
    lines.append("AlphaGenome validation Pearson R (best epoch, no_shift Boda heads):")
    lines.append("")
    lines.append("| Head | Val Pearson R |")
    lines.append("|------|--------------|")
    for head, r in ag_val.get("val_pearson_by_head", {}).items():
        lines.append(f"| {head} | {r:.4f} |")
    if ag_val.get("val_pearson_by_head"):
        best_ag = max(ag_val["val_pearson_by_head"].values())
        lines.append("")
        lines.append(f"**Best AlphaGenome val Pearson:** {best_ag:.4f}.")
    lines.append("")
    lines.append(
        "Malinois validation (from boda2 notebook): Pearson R **0.9131** (K562 val chr 19, 21, X)."
    )
    lines.append("")

    # HashFrag: optional (we compare on original K562 chrom test only)
    hashfrag_m = malinois.get("hashfrag")
    if hashfrag_m:
        lines.append("## 4. HashFrag test sets (optional, not used for primary comparison)")
        lines.append("")
        lines.append("### 4.1 Malinois (boda2 tutorial)")
        lines.append("")
        lines.append("| Set | Pearson R | n |")
        lines.append("|-----|-----------|---|")
        for key in ("in_distribution", "snv_abs", "snv_delta", "ood"):
            v = hashfrag_m.get(key, {})
            lines.append(f"| {key} | {v.get('pearson_r', 0):.4f} | {v.get('n', 0)} |")
        lines.append("")

    if ag_hashfrag and isinstance(ag_hashfrag, dict):
        lines.append("### 4.2 AlphaGenome (per-head HashFrag)")
        lines.append("")
        # ag_hashfrag: { "boda_sum": { "in_distribution": { "pearson_r": ... }, ... }, ... }
        heads = sorted(ag_hashfrag.keys())
        lines.append("| Head | ID | SNV_abs | SNV_delta | OOD |")
        lines.append("|------|----|---------|-----------|-----|")
        for head in heads:
            h = ag_hashfrag[head]
            id_r = h.get("in_distribution", {}).get("pearson_r", 0)
            sa_r = h.get("snv_abs", {}).get("pearson_r", 0)
            sd_r = h.get("snv_delta", {}).get("pearson_r", 0)
            ood_r = h.get("ood", {}).get("pearson_r", 0)
            lines.append(f"| {head} | {id_r:.4f} | {sa_r:.4f} | {sd_r:.4f} | {ood_r:.4f} |")
        lines.append("")

        if hashfrag_m:
            lines.append("### 4.3 Summary")
            lines.append("")
            lines.append("| Metric | Malinois | AlphaGenome (best) | AlphaGenome (worst) |")
            lines.append("|--------|----------|--------------------|---------------------|")
            for key, label in (
                ("in_distribution", "ID"),
                ("snv_abs", "SNV_abs"),
                ("snv_delta", "SNV_delta"),
                ("ood", "OOD"),
            ):
                m_r = hashfrag_m.get(key, {}).get("pearson_r")
                if m_r is None:
                    continue
                ag_vals = [ag_hashfrag[h].get(key, {}).get("pearson_r") for h in heads]
                ag_vals = [x for x in ag_vals if x is not None]
                best_ag = max(ag_vals) if ag_vals else 0
                worst_ag = min(ag_vals) if ag_vals else 0
                lines.append(f"| {label} | {m_r:.4f} | {best_ag:.4f} | {worst_ag:.4f} |")
    if not hashfrag_m:
        lines.append("## 4. HashFrag")
        lines.append("")
        lines.append(
            "Not used. Comparison is on the original K562 chromosome test split (chr 7, 13) only."
        )
        lines.append("")

    lines.append("## 5. Notes")
    lines.append("")
    lines.append(
        "- **Malinois** is evaluated with the boda2 tutorial protocol on the **original K562 test set** (chromosome split chr 7, 13 from DATA-Table_S2), not HashFrag TSVs."
    )
    lines.append(
        "- **AlphaGenome** val numbers are from no_shift training (50% random RC) on chr 19, 21, X."
    )
    lines.append("- Test (chr 7, 13) is held-out; val (chr 19, 21, X) is used for early stopping.")
    lines.append("")

    report = "\n".join(lines)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Wrote report to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
