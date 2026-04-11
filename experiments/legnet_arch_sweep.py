#!/usr/bin/env python
"""LegNet architecture sweep: kernel size, depth, and width.

Tests LegNet architecture variants across training set sizes to find
optimal configurations at each data scale.

Sweeps:
  1. Kernel size: ks=3, 5, 7, 9 (at n=32K, 160K, 296K)
  2. Depth: 4, 6, 8, 10 blocks (at n=32K, 160K, 296K)
  3. Width at scale: narrow/default/wide (at n=160K, 296K)

Each config is trained with 3 seeds and evaluated on the chr-split
test set (real labels, chr7+13).

Usage::

    # Single config test
    python experiments/legnet_arch_sweep.py \\
        --sweep kernel --sizes 32000 --seeds 1 --dry-run

    # Full kernel sweep
    python experiments/legnet_arch_sweep.py \\
        --sweep kernel --sizes 32000 160000 296000

    # Full depth sweep
    python experiments/legnet_arch_sweep.py \\
        --sweep depth --sizes 32000 160000 296000

    # Width sweep at large N
    python experiments/legnet_arch_sweep.py \\
        --sweep width --sizes 160000 296000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture configurations
# ---------------------------------------------------------------------------

# Kernel size sweep: vary ks with default block_sizes
KERNEL_CONFIGS = {
    "ks3": {"block_sizes": [256, 256, 128, 128, 64, 64, 32, 32], "ks": 3},
    "ks5": {"block_sizes": [256, 256, 128, 128, 64, 64, 32, 32], "ks": 5},
    "ks7": {"block_sizes": [256, 256, 128, 128, 64, 64, 32, 32], "ks": 7},
    "ks9": {"block_sizes": [256, 256, 128, 128, 64, 64, 32, 32], "ks": 9},
}

# Depth sweep: vary number of blocks with consistent width schedule
# For each depth, use a downward taper from 256 to 32
DEPTH_CONFIGS = {
    "depth4": {"block_sizes": [256, 128, 64, 32], "ks": 5},
    "depth6": {"block_sizes": [256, 256, 128, 64, 64, 32], "ks": 5},
    "depth8": {"block_sizes": [256, 256, 128, 128, 64, 64, 32, 32], "ks": 5},  # = default
    "depth10": {"block_sizes": [256, 256, 256, 128, 128, 64, 64, 64, 32, 32], "ks": 5},
}

# Width sweep: scale all channel counts
WIDTH_CONFIGS = {
    "narrow": {"block_sizes": [128, 128, 64, 64, 32, 32, 16, 16], "ks": 5},
    "default": {"block_sizes": [256, 256, 128, 128, 64, 64, 32, 32], "ks": 5},
    "wide": {"block_sizes": [512, 512, 256, 256, 128, 128, 64, 64], "ks": 5},
    "xwide": {"block_sizes": [512, 512, 512, 256, 256, 128, 128, 64], "ks": 5},
}

SWEEP_CONFIGS = {
    "kernel": KERNEL_CONFIGS,
    "depth": DEPTH_CONFIGS,
    "width": WIDTH_CONFIGS,
    "all": {**KERNEL_CONFIGS, **DEPTH_CONFIGS, **WIDTH_CONFIGS},
}

# Training HP (fixed: best from prior experiments)
DEFAULT_LR = 0.001
DEFAULT_BS = 512
DEFAULT_EPOCHS = 80
DEFAULT_PATIENCE = 10

DEFAULT_SIZES = [32000, 160000, 296000]


@dataclass
class SweepResult:
    arch_name: str
    n_train: int
    seed: int
    block_sizes: list[int]
    ks: int
    n_params: int
    val_pearson: float
    test_in_dist_pearson: float
    test_snv_abs_pearson: float
    test_snv_delta_pearson: float
    test_ood_pearson: float
    wall_seconds: float


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one_config(
    arch_name: str,
    block_sizes: list[int],
    ks: int,
    n_train: int,
    seed: int,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = DEFAULT_LR,
    bs: int = DEFAULT_BS,
    dry_run: bool = False,
) -> SweepResult | None:
    """Train one architecture config and return results."""
    import torch

    from data.k562 import K562Dataset
    from models.legnet import LegNet, one_hot_encode_batch
    from models.legnet_student import LegNetStudent, TrainConfig

    run_dir = output_dir / arch_name / f"n{n_train}" / f"seed{seed}"
    result_path = run_dir / "result.json"

    # Skip if already done
    if result_path.exists():
        logger.info(f"  SKIP {arch_name} n={n_train} seed={seed}: already done")
        try:
            data = json.loads(result_path.read_text())
            return SweepResult(**{k: data[k] for k in SweepResult.__dataclass_fields__})
        except Exception:
            return None

    if dry_run:
        # Just count params
        model = LegNet(in_channels=4, block_sizes=block_sizes, ks=ks, task_mode="k562")
        n_params = count_params(model)
        logger.info(
            f"  [DRY] {arch_name}: blocks={block_sizes}, ks={ks}, "
            f"params={n_params:,}, n_train={n_train}"
        )
        return None

    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"  Training {arch_name}: blocks={block_sizes}, ks={ks}, n_train={n_train}, seed={seed}"
    )

    t0 = time.time()

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load training data (chr-split: train on all except chr7+13)
    ds_train = K562Dataset(
        data_path=str(REPO / "data" / "k562"),
        split="train",
        label_column="K562_log2FC",
        use_chromosome_fallback=True,
    )
    all_seqs = list(ds_train.sequences)
    all_labels = ds_train.labels.astype(np.float32)

    # Subsample
    rng = np.random.RandomState(seed)
    if n_train < len(all_seqs):
        idx = rng.choice(len(all_seqs), size=n_train, replace=False)
        train_seqs = [all_seqs[i] for i in idx]
        train_labels = all_labels[idx]
    else:
        train_seqs = all_seqs
        train_labels = all_labels

    # Validation set (chr-split val = chr6)
    ds_val = K562Dataset(
        data_path=str(REPO / "data" / "k562"),
        split="val",
        label_column="K562_log2FC",
        use_chromosome_fallback=True,
    )
    val_seqs = list(ds_val.sequences)
    val_labels = ds_val.labels.astype(np.float32)

    # Train
    student = LegNetStudent(
        in_channels=4,
        sequence_length=200,
        task_mode="k562",
        ensemble_size=1,
        block_sizes=block_sizes,
        ks=ks,
        train_config=TrainConfig(
            batch_size=bs,
            lr=lr,
            epochs=epochs,
            early_stopping_patience=patience,
        ),
    )
    student.fit(train_seqs, train_labels, val_sequences=val_seqs, val_labels=val_labels)
    n_params = count_params(student.models[0])

    # Validation Pearson
    val_preds = student.predict(val_seqs)
    val_r = float(np.corrcoef(val_preds, val_labels)[0, 1])
    if np.isnan(val_r):
        val_r = 0.0

    # Test evaluation (chr7+13)
    test_metrics = {}

    # In-dist test
    try:
        ds_test = K562Dataset(
            data_path=str(REPO / "data" / "k562"),
            split="test",
            label_column="K562_log2FC",
            use_chromosome_fallback=True,
        )
        test_seqs = list(ds_test.sequences)
        test_labels = ds_test.labels.astype(np.float32)
        test_preds = student.predict(test_seqs)
        mask = np.isfinite(test_labels)
        r = float(np.corrcoef(test_preds[mask], test_labels[mask])[0, 1])
        test_metrics["in_dist_pearson"] = r if not np.isnan(r) else 0.0
    except Exception as e:
        logger.error(f"    In-dist test failed: {e}")
        test_metrics["in_dist_pearson"] = 0.0

    # SNV test
    import pandas as pd

    snv_path = REPO / "data" / "k562" / "test_sets" / "test_snv_pairs_hashfrag.tsv"
    test_metrics["snv_abs_pearson"] = 0.0
    test_metrics["snv_delta_pearson"] = 0.0
    if snv_path.exists():
        try:
            snv_df = pd.read_csv(snv_path, sep="\t")
            # Filter to chr7+13
            if "IDs_ref" in snv_df.columns:
                chroms = snv_df["IDs_ref"].str.split(":", expand=True)[0]
                snv_df = snv_df[chroms.isin({"7", "13", "chr7", "chr13"})].reset_index(drop=True)
            ref_preds = student.predict(snv_df["sequence_ref"].tolist())
            alt_preds = student.predict(snv_df["sequence_alt"].tolist())
            if "K562_log2FC_alt" in snv_df.columns:
                alt_true = snv_df["K562_log2FC_alt"].to_numpy(dtype=np.float32)
                m = np.isfinite(alt_true)
                if m.sum() > 0:
                    r = float(np.corrcoef(alt_preds[m], alt_true[m])[0, 1])
                    test_metrics["snv_abs_pearson"] = r if not np.isnan(r) else 0.0
            if "delta_log2FC" in snv_df.columns:
                delta_true = snv_df["delta_log2FC"].to_numpy(dtype=np.float32)
                delta_pred = alt_preds - ref_preds
                m = np.isfinite(delta_true)
                if m.sum() > 0:
                    r = float(np.corrcoef(delta_pred[m], delta_true[m])[0, 1])
                    test_metrics["snv_delta_pearson"] = r if not np.isnan(r) else 0.0
        except Exception as e:
            logger.warning(f"    SNV test failed: {e}")

    # OOD test
    ood_path = REPO / "data" / "k562" / "test_sets" / "test_ood_designed_k562.tsv"
    test_metrics["ood_pearson"] = 0.0
    if ood_path.exists():
        try:
            ood_df = pd.read_csv(ood_path, sep="\t")
            ood_preds = student.predict(ood_df["sequence"].tolist())
            ood_true = ood_df["K562_log2FC"].to_numpy(dtype=np.float32)
            m = np.isfinite(ood_true)
            if m.sum() > 0:
                r = float(np.corrcoef(ood_preds[m], ood_true[m])[0, 1])
                test_metrics["ood_pearson"] = r if not np.isnan(r) else 0.0
        except Exception as e:
            logger.warning(f"    OOD test failed: {e}")

    wall = time.time() - t0

    result = SweepResult(
        arch_name=arch_name,
        n_train=n_train,
        seed=seed,
        block_sizes=block_sizes,
        ks=ks,
        n_params=n_params,
        val_pearson=val_r,
        test_in_dist_pearson=test_metrics.get("in_dist_pearson", 0.0),
        test_snv_abs_pearson=test_metrics.get("snv_abs_pearson", 0.0),
        test_snv_delta_pearson=test_metrics.get("snv_delta_pearson", 0.0),
        test_ood_pearson=test_metrics.get("ood_pearson", 0.0),
        wall_seconds=wall,
    )

    # Save
    from dataclasses import asdict

    result_path.write_text(json.dumps(asdict(result), indent=2))
    logger.info(
        f"    {arch_name} n={n_train} seed={seed}: "
        f"val={val_r:.4f} test_id={test_metrics.get('in_dist_pearson', 0):.4f} "
        f"ood={test_metrics.get('ood_pearson', 0):.4f} "
        f"params={n_params:,} time={wall:.0f}s"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="LegNet architecture sweep")
    parser.add_argument(
        "--sweep",
        choices=["kernel", "depth", "width", "all"],
        default="all",
        help="Which sweep to run.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Training sizes to test.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds (42, 1042, 2042, ...).",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--bs", type=int, default=DEFAULT_BS, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max epochs.")
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE, help="Early stop patience."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/legnet_arch_sweep).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print configs and param counts.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run only a specific config name (e.g., 'ks7' or 'depth6').",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = (
        Path(args.output_dir) if args.output_dir else REPO / "outputs" / "legnet_arch_sweep"
    )
    configs = SWEEP_CONFIGS[args.sweep]

    if args.config:
        if args.config not in configs:
            logger.error(f"Config '{args.config}' not found. Available: {list(configs.keys())}")
            sys.exit(1)
        configs = {args.config: configs[args.config]}

    seeds = [42 + i * 1000 for i in range(args.seeds)]

    logger.info(f"Sweep: {args.sweep}")
    logger.info(f"Configs: {list(configs.keys())}")
    logger.info(f"Sizes: {args.sizes}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {output_dir}")

    all_results = []
    total = len(configs) * len(args.sizes) * len(seeds)
    done = 0

    for arch_name, arch_cfg in configs.items():
        for n_train in args.sizes:
            for seed in seeds:
                done += 1
                logger.info(f"[{done}/{total}] {arch_name} n={n_train} seed={seed}")
                try:
                    result = run_one_config(
                        arch_name=arch_name,
                        block_sizes=arch_cfg["block_sizes"],
                        ks=arch_cfg["ks"],
                        n_train=n_train,
                        seed=seed,
                        output_dir=output_dir,
                        epochs=args.epochs,
                        patience=args.patience,
                        lr=args.lr,
                        bs=args.bs,
                        dry_run=args.dry_run,
                    )
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    logger.error(traceback.format_exc())

    # Print summary table
    if all_results:
        logger.info("\n" + "=" * 90)
        logger.info("SUMMARY")
        logger.info("=" * 90)
        logger.info(
            f"{'Config':<12} {'N':>8} {'Params':>10} "
            f"{'Val r':>7} {'Test ID':>8} {'SNV abs':>8} {'OOD':>7} {'Time':>6}"
        )
        logger.info("-" * 90)

        # Group by (arch, n_train) and show mean +/- std
        from collections import defaultdict

        groups = defaultdict(list)
        for r in all_results:
            groups[(r.arch_name, r.n_train)].append(r)

        for (arch, n), runs in sorted(groups.items()):
            vals = [r.test_in_dist_pearson for r in runs]
            snvs = [r.test_snv_abs_pearson for r in runs]
            oods = [r.test_ood_pearson for r in runs]
            times = [r.wall_seconds for r in runs]
            logger.info(
                f"{arch:<12} {n:>8,} {runs[0].n_params:>10,} "
                f"{np.mean([r.val_pearson for r in runs]):>7.4f} "
                f"{np.mean(vals):>7.4f}+/-{np.std(vals):.3f} "
                f"{np.mean(snvs):>7.4f} "
                f"{np.mean(oods):>6.4f} "
                f"{np.mean(times):>5.0f}s"
            )

        # Save summary JSON
        from dataclasses import asdict

        summary_path = output_dir / f"summary_{args.sweep}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
        logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
