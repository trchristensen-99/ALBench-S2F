#!/usr/bin/env python
"""Evaluate AlphaGenome Boda heads (sum, mean, max, center) on chr 7, 13 test set.

Writes outputs/ag_chrom_test_results.json for direct comparison with Malinois.
Run from repo root (on HPC or locally with checkpoints and data).

  uv run python scripts/analysis/eval_ag_chrom_test.py
  uv run python scripts/analysis/eval_ag_chrom_test.py --data_path data/k562 --output outputs/ag_chrom_test_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval_ag import evaluate_chrom_test

CONFIGS = [
    ("boda_sum", "outputs/ag_sum/best_model", "alphagenome_k562_head_boda_sum_512_512_v4"),
    ("boda_mean", "outputs/ag_mean/best_model", "alphagenome_k562_head_boda_mean_512_512_v4"),
    ("boda_max", "outputs/ag_max/best_model", "alphagenome_k562_head_boda_max_512_512_v4"),
    ("boda_center", "outputs/ag_center/best_model", "alphagenome_k562_head_boda_center_512_512_v4"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default="data/k562", help="K562 data dir (for K562FullDataset).")
    p.add_argument(
        "--output", default="outputs/ag_chrom_test_results.json", help="Output JSON path."
    )
    args = p.parse_args()

    out: dict[str, dict] = {}
    for label, ckpt_dir, head_name in CONFIGS:
        ckpt_path = Path(ckpt_dir).resolve()
        if not (ckpt_path / "checkpoint").exists():
            print(
                f"[eval_ag_chrom_test] Skip {label}: no checkpoint at {ckpt_path}", file=sys.stderr
            )
            continue
        print(f"[eval_ag_chrom_test] Evaluating {label} on chr 7, 13 ...", file=sys.stderr)
        out[label] = evaluate_chrom_test(str(ckpt_path), head_name, data_path=args.data_path)

    if not out:
        print("[eval_ag_chrom_test] No checkpoints found.", file=sys.stderr)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval_ag_chrom_test] Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
