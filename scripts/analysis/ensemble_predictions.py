#!/usr/bin/env python3
"""Ensemble predictions across seeds and compute metrics.

For each model x cell configuration, loads test_predictions.npz from multiple
seed directories, averages predictions across seeds, and computes Pearson R,
Spearman R, and MSE on the ensembled predictions.

Usage:
    python scripts/analysis/ensemble_predictions.py
    python scripts/analysis/ensemble_predictions.py --output-root outputs
    python scripts/analysis/ensemble_predictions.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry: (display_name, cell) -> list of directories to glob for
# test_predictions.npz.  If a path is a parent directory, we auto-discover
# seed subdirectories containing the npz file.
ENSEMBLE_CONFIGS: dict[tuple[str, str], list[str]] = {
    # AlphaGenome Stage 2
    ("AG all-folds S2", "k562"): [
        "outputs/stage2_k562_full_train",
    ],
    # Enformer S1
    ("Enformer S1", "k562"): [
        "outputs/enformer_k562_3seeds",
    ],
    # Borzoi S1
    ("Borzoi S1", "k562"): [
        "outputs/borzoi_k562_3seeds",
    ],
    # NT v2 S1
    ("NT v2 S1", "k562"): [
        "outputs/nt_k562_3seeds",
    ],
    # DREAM-RNN (3-member ensemble per seed)
    ("DREAM-RNN", "k562"): [
        "outputs/dream_rnn_k562_3seeds",
    ],
    ("DREAM-RNN", "hepg2"): [
        "outputs/dream_rnn_hepg2_3seeds",
    ],
    ("DREAM-RNN", "sknsh"): [
        "outputs/dream_rnn_sknsh_3seeds",
    ],
    # DREAM-RNN single model (ensemble_size=1)
    ("DREAM-RNN single", "k562"): [
        "outputs/dream_rnn_k562_single",
    ],
    ("DREAM-RNN single", "hepg2"): [
        "outputs/dream_rnn_hepg2_single",
    ],
    ("DREAM-RNN single", "sknsh"): [
        "outputs/dream_rnn_sknsh_single",
    ],
    # Malinois
    ("Malinois", "k562"): [
        "outputs/malinois_k562_3seeds",
    ],
}

# Prediction keys: (pred_key, true_key, metric_name)
PREDICTION_KEYS = [
    ("in_dist_pred", "in_dist_true", "in_distribution"),
    ("snv_alt_pred", "snv_alt_true", "snv_abs"),
    ("snv_delta_pred", "snv_delta_true", "snv_delta"),
    ("ood_pred", "ood_true", "ood"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_corr(pred: np.ndarray, target: np.ndarray, fn) -> float:
    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.std(pred) == 0.0 or np.std(target) == 0.0:
        return 0.0
    return float(fn(pred, target)[0])


def _discover_npz_files(search_dirs: list[str], root: Path) -> list[Path]:
    """Find all test_predictions.npz files under the given directories."""
    npz_files: list[Path] = []
    for d in search_dirs:
        base = root / d
        if not base.exists():
            continue
        # Check if the directory itself contains the npz
        direct = base / "test_predictions.npz"
        if direct.exists():
            npz_files.append(direct)
            continue
        # Otherwise glob for subdirectories (seed_*, run_*, etc.)
        found = sorted(base.glob("**/test_predictions.npz"))
        npz_files.extend(found)
    return npz_files


def _compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": _safe_corr(pred, true, pearsonr),
        "spearman_r": _safe_corr(pred, true, spearmanr),
        "mse": float(np.mean((pred - true) ** 2)),
    }


def ensemble_and_evaluate(
    npz_files: list[Path],
) -> dict[str, dict[str, float]]:
    """Average predictions across seeds and compute metrics."""
    # Load all npz files
    all_data = [np.load(f) for f in npz_files]
    n_seeds = len(all_data)

    metrics: dict[str, dict[str, float]] = {}
    for pred_key, true_key, metric_name in PREDICTION_KEYS:
        # Check if key exists in all files
        if not all(pred_key in d for d in all_data):
            continue
        if not all(true_key in d for d in all_data):
            continue

        # Get true values from first seed
        true_vals = all_data[0][true_key]

        # Average predictions across seeds
        preds = np.stack([d[pred_key] for d in all_data], axis=0)
        ensemble_pred = np.mean(preds, axis=0)

        metrics[metric_name] = _compute_metrics(ensemble_pred, true_vals)

    # Also store per-seed metrics for reference
    per_seed: dict[str, list[dict[str, float]]] = {}
    for pred_key, true_key, metric_name in PREDICTION_KEYS:
        if not all(pred_key in d for d in all_data):
            continue
        if not all(true_key in d for d in all_data):
            continue
        per_seed[metric_name] = []
        for d in all_data:
            per_seed[metric_name].append(_compute_metrics(d[pred_key], d[true_key]))

    # Close npz files
    for d in all_data:
        d.close()

    return {
        "ensemble": metrics,
        "per_seed": per_seed,
        "n_seeds": n_seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble predictions across seeds")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for outputs (default: outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show discovered files, don't compute metrics",
    )
    args = parser.parse_args()

    root = args.output_root.parent if args.output_root.name == "outputs" else args.output_root
    if (root / "outputs").is_dir():
        root = root  # root is project root
    else:
        root = Path(".")

    summary_rows: list[dict] = []

    for (display_name, cell), search_dirs in ENSEMBLE_CONFIGS.items():
        npz_files = _discover_npz_files(search_dirs, root)

        if not npz_files:
            print(f"[SKIP] {display_name} ({cell}): no test_predictions.npz found")
            continue

        print(f"\n{'=' * 60}")
        print(f"{display_name} ({cell}): {len(npz_files)} seed(s)")
        for f in npz_files:
            print(f"  - {f}")

        if args.dry_run:
            continue

        if len(npz_files) < 2:
            print("  WARNING: only 1 seed found, no ensembling needed")

        result = ensemble_and_evaluate(npz_files)

        # Print results
        print(f"  Ensemble metrics ({result['n_seeds']} seeds):")
        for metric_name, vals in result["ensemble"].items():
            print(
                f"    {metric_name:20s}: "
                f"Pearson={vals['pearson_r']:.4f}  "
                f"Spearman={vals['spearman_r']:.4f}  "
                f"MSE={vals['mse']:.4f}"
            )

        # Print per-seed for comparison
        print(f"  Per-seed Pearson R:")
        for metric_name, seed_vals in result["per_seed"].items():
            seed_rs = [v["pearson_r"] for v in seed_vals]
            seed_str = ", ".join(f"{r:.4f}" for r in seed_rs)
            print(f"    {metric_name:20s}: [{seed_str}]")

        # Save results
        safe_name = display_name.lower().replace(" ", "_").replace("-", "_")
        out_dir = root / "outputs" / f"{safe_name}_{cell}_ensemble"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "test_metrics.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Saved: {out_path}")

        # Collect for summary table
        if "in_distribution" in result["ensemble"]:
            summary_rows.append(
                {
                    "model": display_name,
                    "cell": cell,
                    "n_seeds": result["n_seeds"],
                    "in_dist_pearson": result["ensemble"]["in_distribution"]["pearson_r"],
                    "snv_abs_pearson": result["ensemble"]
                    .get("snv_abs", {})
                    .get("pearson_r", float("nan")),
                    "snv_delta_pearson": result["ensemble"]
                    .get("snv_delta", {})
                    .get("pearson_r", float("nan")),
                    "ood_pearson": result["ensemble"].get("ood", {}).get("pearson_r", float("nan")),
                }
            )

    # Print summary table
    if summary_rows and not args.dry_run:
        print(f"\n{'=' * 80}")
        print("SUMMARY TABLE")
        print(f"{'=' * 80}")
        header = (
            f"{'Model':<25s} {'Cell':<8s} {'Seeds':>5s} "
            f"{'InDist':>8s} {'SNV_abs':>8s} {'SNV_d':>8s} {'OOD':>8s}"
        )
        print(header)
        print("-" * len(header))
        for row in summary_rows:
            print(
                f"{row['model']:<25s} {row['cell']:<8s} {row['n_seeds']:>5d} "
                f"{row['in_dist_pearson']:>8.4f} {row['snv_abs_pearson']:>8.4f} "
                f"{row['snv_delta_pearson']:>8.4f} {row['ood_pearson']:>8.4f}"
            )


if __name__ == "__main__":
    main()
