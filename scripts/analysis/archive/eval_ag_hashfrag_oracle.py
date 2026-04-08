#!/usr/bin/env python
"""Evaluate all 10 hashFrag oracle seeds and aggregate results.

Loads test_metrics.json from each seed (written by train_oracle_alphagenome_hashfrag.py)
if available, otherwise runs full evaluation from checkpoint.

Writes outputs/ag_hashfrag_oracle_results.json with per-seed and summary statistics.

Usage:
  uv run python scripts/analysis/eval_ag_hashfrag_oracle.py
  uv run python scripts/analysis/eval_ag_hashfrag_oracle.py --output_dir outputs/ag_hashfrag_oracle --data_path data/k562
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

N_ORACLES = 10
HEAD_NAME = "alphagenome_k562_head_hashfrag_boda_flatten_512_512_v4"
TEST_SETS = ["in_distribution", "snv_abs", "snv_delta", "ood"]
METRICS = ["pearson_r", "spearman_r", "mse"]


def _load_or_eval(oracle_dir: Path, data_path: str) -> dict | None:
    """Load cached test_metrics.json or run eval from checkpoint."""
    json_path = oracle_dir / "test_metrics.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return data.get("test_metrics", data)

    ckpt_path = oracle_dir / "best_model"
    if not (ckpt_path / "checkpoint").exists():
        print(f"[eval] Skip {oracle_dir.name}: no checkpoint found", file=sys.stderr)
        return None

    print(f"[eval] Running eval for {oracle_dir.name} …", file=sys.stderr)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval_ag import evaluate_hashfrag_test_sets_600bp

    return evaluate_hashfrag_test_sets_600bp(
        ckpt_dir=str(ckpt_path),
        head_name=HEAD_NAME,
        data_path=data_path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", default="outputs/ag_hashfrag_oracle")
    p.add_argument("--data_path", default="data/k562")
    p.add_argument("--output", default="outputs/ag_hashfrag_oracle_results.json")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    all_results: dict[str, dict] = {}

    for i in range(N_ORACLES):
        oracle_dir = output_dir / f"oracle_{i}"
        metrics = _load_or_eval(oracle_dir, args.data_path)
        if metrics is not None:
            all_results[str(i)] = metrics
        else:
            print(f"[eval] No results for oracle_{i}", file=sys.stderr)

    if not all_results:
        print("[eval] No seed results found.", file=sys.stderr)
        sys.exit(1)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for test_set in TEST_SETS:
        summary[test_set] = {}
        for metric in METRICS:
            vals = [
                all_results[s][test_set][metric]
                for s in all_results
                if test_set in all_results[s] and metric in all_results[s][test_set]
            ]
            if vals:
                summary[test_set][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_seeds": len(vals),
                    "per_seed": vals,
                }

    output = {
        "per_seed": all_results,
        "summary": summary,
        "seeds": list(all_results.keys()),
        "n_seeds": len(all_results),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[eval] Wrote {out_path}", file=sys.stderr)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'Test set':<20} {'Metric':<14} {'Mean':>8} {'±Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)
    for test_set in TEST_SETS:
        if test_set not in summary:
            continue
        for metric in METRICS:
            if metric not in summary[test_set]:
                continue
            s = summary[test_set][metric]
            print(
                f"{test_set:<20} {metric:<14} {s['mean']:>8.4f} {s['std']:>8.4f} "
                f"{s['min']:>8.4f} {s['max']:>8.4f}"
            )
        print()


if __name__ == "__main__":
    main()
