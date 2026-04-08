#!/usr/bin/env python
"""Export AlphaGenome HashFrag test metrics to a single JSON for comparison with Malinois.

Run from repo root when outputs/ag_*/best_model checkpoints exist. Writes
outputs/ag_hashfrag_results.json so compare_malinois_alphagenome_results.py can use it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Same configs as eval_boda_k562
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


def main():
    from eval_ag import evaluate

    out: dict[str, dict] = {}
    for label, ckpt_dir, head_name in CONFIGS:
        ckpt_path = Path(ckpt_dir).resolve()
        if not (ckpt_path / "checkpoint").exists():
            print(
                f"[export_ag_hashfrag] Skip {label}: no checkpoint at {ckpt_path}", file=sys.stderr
            )
            continue
        print(f"[export_ag_hashfrag] Evaluating {label} ...", file=sys.stderr)
        metrics = evaluate(str(ckpt_path), head_name)
        out[label] = {
            k: {"pearson_r": v.get("pearson_r"), "n": v.get("n")}
            for k, v in metrics.items()
            if isinstance(v, dict)
        }

    if not out:
        print("[export_ag_hashfrag] No checkpoints found; not writing file.", file=sys.stderr)
        return

    out_path = Path("outputs/ag_hashfrag_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[export_ag_hashfrag] Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
