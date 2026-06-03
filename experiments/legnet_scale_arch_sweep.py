#!/usr/bin/env python
"""LegNet architecture sweep at large N (1M+) using oracle pseudo-labels.

Tests whether larger model capacity is needed at 1M-2M training examples.
Trains on oracle-labeled random reservoir pool and evaluates on chr-split
test set (chr7+13, real labels).

Configs tested:
  wide      : [512,512,256,256,128,128,64,64], ks=5  (~10.5M params)
  wide_ks7  : [512,512,256,256,128,128,64,64], ks=7  (~13M params)
  depth10   : [512,512,512,256,256,128,128,64,64,64], ks=5  (~20M params, 10 blocks)
  default_ks7: [256,256,128,128,64,64,32,32], ks=7  (~3.2M params)

All configs are compared to existing `default` (ks=5, 2.6M) baseline results
already in outputs/legnet_arch_sweep/default/.

Outputs: outputs/legnet_arch_sweep/{config}/n{N}/seed{seed}/result.json

Usage::

    # Quick test
    python experiments/legnet_scale_arch_sweep.py --config wide --sizes 296000 --seeds 1

    # Full 1M sweep
    python experiments/legnet_scale_arch_sweep.py --config wide --sizes 1000000

    # All 4 configs at 1M
    python experiments/legnet_scale_arch_sweep.py --sizes 1000000

    # Round 2: 2M training
    python experiments/legnet_scale_arch_sweep.py --config wide --sizes 2000000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture configurations for large-N sweep
# ---------------------------------------------------------------------------

LARGE_N_CONFIGS = {
    # Default baseline for reference (already in outputs/legnet_arch_sweep/default/)
    "default": {
        "block_sizes": [256, 256, 128, 128, 64, 64, 32, 32],
        "ks": 5,
        "desc": "baseline 2.6M, ks=5",
    },
    # Wide: same taper as default but 2x channels => ~10.5M params
    "wide": {
        "block_sizes": [512, 512, 256, 256, 128, 128, 64, 64],
        "ks": 5,
        "desc": "wide 10.5M, ks=5",
    },
    # Wide + larger kernel (ks=7 was best at 160K-296K in kernel sweep)
    "wide_ks7": {
        "block_sizes": [512, 512, 256, 256, 128, 128, 64, 64],
        "ks": 7,
        "desc": "wide 13M, ks=7",
    },
    # More depth: 10 blocks at wide channel sizes
    "depth10_wide": {
        "block_sizes": [512, 512, 512, 256, 256, 128, 128, 64, 64, 64],
        "ks": 5,
        "desc": "depth10 wide ~20M, ks=5",
    },
    # Default width but larger kernel (cheaper than wide, might benefit from more data)
    "default_ks7": {
        "block_sizes": [256, 256, 128, 128, 64, 64, 32, 32],
        "ks": 7,
        "desc": "default 3.2M, ks=7",
    },
}

# Training hyperparameters
DEFAULT_LR = 0.001
DEFAULT_BS = 512
DEFAULT_SEED = 42

# Epoch budget: fewer epochs for larger N (rely on early stopping)
# At n=1M, 1 epoch = ~1953 steps at bs=512 => OneCycleLR needs total_steps
EPOCH_BUDGET = {
    296_000: 50,
    500_000: 40,
    1_000_000: 30,
    2_000_000: 20,
}
PATIENCE = 8  # early stopping

POOL_PATH = REPO / "outputs" / "labeled_pools_2m" / "k562" / "ag_s2" / "random" / "pool.npz"


@dataclass
class SweepResult:
    arch_name: str
    n_train: int
    seed: int
    block_sizes: list
    ks: int
    n_params: int
    val_pearson: float
    test_in_dist_pearson: float
    test_snv_abs_pearson: float
    test_snv_delta_pearson: float
    test_ood_pearson: float
    wall_seconds: float
    data_source: str = "oracle_pool"


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_one_config(
    arch_name: str,
    block_sizes: list[int],
    ks: int,
    n_train: int,
    seed: int,
    output_dir: Path,
    epochs: int | None = None,
    patience: int = PATIENCE,
    lr: float = DEFAULT_LR,
    bs: int = DEFAULT_BS,
    dry_run: bool = False,
) -> SweepResult | None:
    """Train one architecture config on oracle pool data and return results."""
    import torch

    from models.legnet import LegNet
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
        model = LegNet(in_channels=4, block_sizes=block_sizes, ks=ks, task_mode="k562")
        n_params = count_params(model)
        logger.info(
            f"  [DRY] {arch_name}: blocks={block_sizes}, ks={ks}, "
            f"params={n_params:,}, n_train={n_train}"
        )
        return None

    run_dir.mkdir(parents=True, exist_ok=True)

    # Determine epochs
    if epochs is None:
        epochs = EPOCH_BUDGET.get(n_train, 30)
    logger.info(
        f"  Training {arch_name}: blocks={block_sizes}, ks={ks}, "
        f"n_train={n_train}, seed={seed}, epochs={epochs}"
    )

    t0 = time.time()

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load training data from oracle pool
    from scripts.generate_labeled_pools import load_pool_subset

    logger.info(f"  Loading {n_train:,} examples from oracle pool: {POOL_PATH}")
    train_seqs, train_labels = load_pool_subset(POOL_PATH, n_train, seed=seed)
    train_labels = train_labels.astype(np.float32)
    logger.info(f"  Loaded {len(train_seqs):,} training sequences")

    # Validation set from chr-split (real labels, chr6)
    from data.k562 import K562Dataset

    ds_val = K562Dataset(
        data_path=str(REPO / "data" / "k562"),
        split="val",
        label_column="K562_log2FC",
    )
    val_seqs = list(ds_val.sequences)
    val_labels = ds_val.labels.astype(np.float32)
    logger.info(f"  Validation set: {len(val_seqs):,} sequences")

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

    # Test evaluation
    test_metrics = {}

    # In-dist test (chr7+13, real labels)
    try:
        ds_test = K562Dataset(
            data_path=str(REPO / "data" / "k562"),
            split="test",
            label_column="K562_log2FC",
            )
        test_seqs = list(ds_test.sequences)
        test_labels_real = ds_test.labels.astype(np.float32)
        test_preds = student.predict(test_seqs)
        mask = np.isfinite(test_labels_real)
        r = float(np.corrcoef(test_preds[mask], test_labels_real[mask])[0, 1])
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
        data_source="oracle_pool",
    )

    result_path.write_text(json.dumps(asdict(result), indent=2))
    logger.info(
        f"    DONE {arch_name} n={n_train} seed={seed}: "
        f"val={val_r:.4f} test_id={test_metrics.get('in_dist_pearson', 0):.4f} "
        f"ood={test_metrics.get('ood_pearson', 0):.4f} "
        f"params={n_params:,} time={wall / 60:.1f}min"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="LegNet architecture sweep at large N using oracle pool labels"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config name (default: run all). Options: " + ", ".join(LARGE_N_CONFIGS),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1_000_000],
        help="Training sizes (default: 1000000)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of seeds (42, 1042, ...). Default: 1",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--bs", type=int, default=DEFAULT_BS)
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO / "outputs" / "legnet_arch_sweep"),
        help="Output directory (default: outputs/legnet_arch_sweep)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)

    configs = LARGE_N_CONFIGS
    if args.config:
        if args.config not in configs:
            logger.error(f"Config '{args.config}' not found. Options: {list(configs.keys())}")
            sys.exit(1)
        configs = {args.config: configs[args.config]}

    seeds = [42 + i * 1000 for i in range(args.seeds)]

    logger.info(f"Configs: {list(configs.keys())}")
    logger.info(f"Sizes: {args.sizes}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Pool: {POOL_PATH}")

    all_results = []
    total = len(configs) * len(args.sizes) * len(seeds)
    done = 0

    for arch_name, arch_cfg in configs.items():
        for n_train in args.sizes:
            for seed in seeds:
                done += 1
                logger.info(
                    f"[{done}/{total}] {arch_name} ({arch_cfg['desc']}) n={n_train:,} seed={seed}"
                )
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
                    logger.error(f"  FAILED {arch_name} n={n_train}: {e}")
                    logger.error(traceback.format_exc())

    # Summary
    if all_results:
        logger.info("\n" + "=" * 100)
        logger.info("SUMMARY")
        logger.info("=" * 100)
        logger.info(
            f"{'Config':<14} {'N':>10} {'Params':>12} "
            f"{'Val r':>7} {'Test ID':>8} {'SNV abs':>8} {'OOD':>7} {'Time':>7}"
        )
        logger.info("-" * 100)

        from collections import defaultdict

        groups = defaultdict(list)
        for r in all_results:
            groups[(r.arch_name, r.n_train)].append(r)

        for (arch, n), runs in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
            logger.info(
                f"{arch:<14} {n:>10,} {runs[0].n_params:>12,}  "
                f"{np.mean([r.val_pearson for r in runs]):>6.4f}  "
                f"{np.mean([r.test_in_dist_pearson for r in runs]):>7.4f}  "
                f"{np.mean([r.test_snv_abs_pearson for r in runs]):>7.4f}  "
                f"{np.mean([r.test_ood_pearson for r in runs]):>6.4f}  "
                f"{np.mean([r.wall_seconds for r in runs]) / 60:>5.1f}m"
            )

        summary_path = output_dir / "summary_large_n.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
        logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
