#!/usr/bin/env python
"""Batch-evaluate Boda AlphaGenome heads on K562 test sets (ID / OOD / SNV).

This is a thin wrapper around ``eval_ag.evaluate`` that:
  * Looks for checkpoints in the standard Boda output dirs:
      - outputs/ag_flatten/best_model
      - outputs/ag_sum/best_model
      - outputs/ag_mean/best_model
      - outputs/ag_max/best_model
      - outputs/ag_center/best_model
  * Uses the v4 head names that match the training script.
  * Prints a compact TSV-style summary table.
"""

from __future__ import annotations

import sys
from pathlib import Path

from eval_ag import evaluate

CONFIGS = [
    (
        "boda_flatten",
        "outputs/ag_flatten/best_model",
        "alphagenome_k562_head_boda_flatten_512_512_v4",
    ),
    ("boda_sum", "outputs/ag_sum/best_model", "alphagenome_k562_head_boda_sum_512_512_v4"),
    ("boda_mean", "outputs/ag_mean/best_model", "alphagenome_k562_head_boda_mean_512_512_v4"),
    ("boda_max", "outputs/ag_max/best_model", "alphagenome_k562_head_boda_max_512_512_v4"),
    ("boda_center", "outputs/ag_center/best_model", "alphagenome_k562_head_boda_center_512_512_v4"),
]


def main() -> None:
    rows: list[tuple[str, float, float, float, float]] = []

    for label, ckpt_dir, head_name in CONFIGS:
        ckpt_path = Path(ckpt_dir).resolve()
        if not (ckpt_path / "checkpoint").exists():
            print(
                f"[eval_boda_k562] Skipping {label}: no checkpoint at {ckpt_path}", file=sys.stderr
            )
            continue

        print(f"[eval_boda_k562] Evaluating {label} from {ckpt_path}", file=sys.stderr)
        metrics = evaluate(str(ckpt_path), head_name)

        rows.append(
            (
                label,
                float(metrics["in_distribution"]["pearson_r"]),
                float(metrics["snv_abs"]["pearson_r"]),
                float(metrics["snv_delta"]["pearson_r"]),
                float(metrics["ood"]["pearson_r"]),
            )
        )

    if not rows:
        print("[eval_boda_k562] No checkpoints found; nothing to report.", file=sys.stderr)
        return

    # Header
    print("head\tID_pearson\tSNV_abs_pearson\tSNV_delta_pearson\tOOD_pearson")
    for label, id_r, snv_abs_r, snv_delta_r, ood_r in rows:
        print(f"{label}\t{id_r:.4f}\t{snv_abs_r:.4f}\t{snv_delta_r:.4f}\t{ood_r:.4f}")


if __name__ == "__main__":
    main()
