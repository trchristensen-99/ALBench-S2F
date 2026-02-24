#!/usr/bin/env python
"""Compare Malinois vs AlphaGenome adapter heads on K562 HashFrag test sets.

Evaluates Malinois and each available Boda head checkpoint on the three
HashFrag test sets (ID / SNV_abs / SNV_delta / OOD) and prints a TSV
comparison table.

Usage (from repo root on HPC after training is complete)::

    python scripts/analysis/compare_malinois_ag.py \\
        --boda_dir ~/boda2-main \\
        --model_path ~/my-model.epoch_5-step_19885.pkl \\
        --test_tsv_dir data/k562/test_sets \\
        --ag_outputs_dir outputs \\
        --output_file results/k562_comparison.tsv

The script skips any AlphaGenome head whose ``best_model/checkpoint`` is not
found (e.g. training hasn't finished yet) and notes this on stderr.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--boda_dir",
        required=True,
        help="Path to the boda2-main repository root.",
    )
    p.add_argument(
        "--model_path",
        required=True,
        help="Path to 'my-model.epoch_5-step_19885.pkl'.",
    )
    p.add_argument(
        "--test_tsv_dir",
        default="data/k562/test_sets",
        help="Directory containing the HashFrag test TSVs (default: data/k562/test_sets).",
    )
    p.add_argument(
        "--ag_outputs_dir",
        default="outputs",
        help="Parent directory containing ag_flatten/, ag_sum/, etc.",
    )
    p.add_argument(
        "--output_file",
        default=None,
        help="If set, write the TSV table to this file in addition to printing.",
    )
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--target_len",
        type=int,
        default=200,
        help="Sequence pad/crop length for Malinois (default: 200 bp).",
    )
    return p.parse_args()


# AlphaGenome head configs: (display_label, output_subdir, unique_head_name)
_AG_HEAD_CONFIGS = [
    ("AlphaGenome-flatten", "ag_flatten", "alphagenome_k562_head_boda_flatten_512_512_v4"),
    ("AlphaGenome-sum", "ag_sum", "alphagenome_k562_head_boda_sum_512_512_v4"),
    ("AlphaGenome-mean", "ag_mean", "alphagenome_k562_head_boda_mean_512_512_v4"),
    ("AlphaGenome-max", "ag_max", "alphagenome_k562_head_boda_max_512_512_v4"),
    ("AlphaGenome-center", "ag_center", "alphagenome_k562_head_boda_center_512_512_v4"),
]


def main() -> None:
    args = get_args()

    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(Path(__file__).parent))

    from eval_malinois_baseline import eval_on_hashfrag

    rows: list[tuple[str, float, float, float, float]] = []

    # ── Malinois ─────────────────────────────────────────────────────────────
    sys.path.insert(0, os.path.abspath(args.boda_dir))
    from boda.model.deprecated_mpra_basset import MPRA_Basset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare] Loading Malinois from {args.model_path} ...", file=sys.stderr)
    malinois = MPRA_Basset(basset_weights_path=args.model_path)
    malinois.to(device)
    malinois.eval()

    mal_m = eval_on_hashfrag(malinois, args.test_tsv_dir, args.batch_size, device, args.target_len)
    rows.append(
        (
            "Malinois",
            float(mal_m["in_distribution"]["pearson_r"]),
            float(mal_m["snv_abs"]["pearson_r"]),
            float(mal_m["snv_delta"]["pearson_r"]),
            float(mal_m["ood"]["pearson_r"]),
        )
    )
    print("[compare] Malinois done.", file=sys.stderr)

    # ── AlphaGenome adapter heads ─────────────────────────────────────────────
    import eval_ag  # noqa: E402 (at repo root)

    ag_dir = Path(args.ag_outputs_dir)
    for label, subdir, head_name in _AG_HEAD_CONFIGS:
        ckpt_path = ag_dir / subdir / "best_model"
        if not (ckpt_path / "checkpoint").exists():
            print(
                f"[compare] Skipping {label}: no checkpoint at {ckpt_path}",
                file=sys.stderr,
            )
            continue
        print(f"[compare] Evaluating {label} ...", file=sys.stderr)
        m = eval_ag.evaluate(str(ckpt_path), head_name)
        rows.append(
            (
                label,
                float(m["in_distribution"]["pearson_r"]),
                float(m["snv_abs"]["pearson_r"]),
                float(m["snv_delta"]["pearson_r"]),
                float(m["ood"]["pearson_r"]),
            )
        )

    # ── Print / save table ────────────────────────────────────────────────────
    header = "model\tID_pearson\tSNV_abs_pearson\tSNV_delta_pearson\tOOD_pearson"
    lines = [header]
    for label, id_r, snv_abs_r, snv_delta_r, ood_r in rows:
        lines.append(f"{label}\t{id_r:.4f}\t{snv_abs_r:.4f}\t{snv_delta_r:.4f}\t{ood_r:.4f}")

    print("\n" + "\n".join(lines))

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines) + "\n")
        print(f"\n[compare] Saved table to {args.output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
