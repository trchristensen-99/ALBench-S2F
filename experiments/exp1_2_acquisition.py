#!/usr/bin/env python
"""Experiment 1.2: Acquisition function benchmarking.

Starting from an initial labeled set, uses different acquisition strategies
to select sequences from a reservoir pool, then retrains the student on the
combined (initial + selected) data and evaluates on the oracle-labeled test
panel.

Usage::

    # Single acquisition function
    uv run python experiments/exp1_2_acquisition.py \
        --task k562 --student dream_rnn \
        --reservoir random --acquisition uncertainty --regime small

    # Multiple acquisition functions in one run
    uv run python experiments/exp1_2_acquisition.py \
        --task k562 --student dream_rnn \
        --reservoir random \
        --acquisition random uncertainty diversity badge \
        --regime small medium

    # Quick smoke test
    uv run python experiments/exp1_2_acquisition.py \
        --task yeast --student dream_rnn \
        --reservoir random --acquisition random \
        --regime small --n-replicates 1 --no-hp-sweep
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Re-use infrastructure from exp1_1
from experiments.exp1_1_scaling import (  # noqa: E402
    HP_GRIDS,
    HP_GRIDS_LARGE_N,
    LARGE_N_THRESHOLD,
    TASK_CONFIGS,
    RunResult,
    _encode_sequences_for_ag,
    _get_ag_model_and_encoder,
    _load_oracle,
    _load_pool_sequences,
    _load_reservoir,
    _train_student,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Acquisition registry
# ---------------------------------------------------------------------------

ACQUISITION_REGISTRY: dict[str, type] = {}


def _populate_registry() -> None:
    """Lazily import acquisition classes so the module loads even if deps
    are missing (e.g. on the login node for config inspection)."""
    if ACQUISITION_REGISTRY:
        return
    from albench.acquisition import (
        BADGEAcquisition,
        BatchBALDAcquisition,
        CombinedAcquisition,
        DiversityAcquisition,
        EnsembleAcquisition,
        PriorKnowledgeAcquisition,
        RandomAcquisition,
        UncertaintyAcquisition,
    )

    ACQUISITION_REGISTRY.update(
        {
            "random": RandomAcquisition,
            "uncertainty": UncertaintyAcquisition,
            "diversity": DiversityAcquisition,
            "badge": BADGEAcquisition,
            "batchbald": BatchBALDAcquisition,
            "combined": CombinedAcquisition,
            "ensemble": EnsembleAcquisition,
            "prior_knowledge": PriorKnowledgeAcquisition,
        }
    )


def _make_acquisition(name: str, seed: int) -> Any:
    """Instantiate an acquisition function by name."""
    _populate_registry()
    cls = ACQUISITION_REGISTRY[name]

    # Only pass seed to classes that accept it
    import inspect

    sig = inspect.signature(cls.__init__)
    if "seed" in sig.parameters:
        return cls(seed=seed)
    return cls()


# ---------------------------------------------------------------------------
# Data regime definitions
# ---------------------------------------------------------------------------

DATA_REGIMES = {
    "small": {"initial_size": 1000, "batch_size": 20000},
    "medium": {"initial_size": 10000, "batch_size": 50000},
    "large": {"initial_size": 50000, "batch_size": 100000},
}

# ---------------------------------------------------------------------------
# HP config builder (mirrors exp1_1)
# ---------------------------------------------------------------------------


def _build_hp_configs(student_type: str, n_train: int, hp_sweep: bool) -> list[dict]:
    """Build HP grid for a given total training size."""
    if not hp_sweep:
        if student_type.startswith("alphagenome"):
            return [{"learning_rate": 1e-3, "batch_size": 128}]
        return [{"learning_rate": 0.005, "batch_size": 1024}]

    if n_train >= LARGE_N_THRESHOLD and student_type in HP_GRIDS_LARGE_N:
        grid = HP_GRIDS_LARGE_N[student_type]
    elif student_type in HP_GRIDS:
        grid = HP_GRIDS[student_type]
    else:
        if student_type.startswith("alphagenome"):
            return [{"learning_rate": 1e-3, "batch_size": 128}]
        return [{"learning_rate": 0.005, "batch_size": 1024}]

    return [dict(zip(grid.keys(), vals)) for vals in itertools.product(*grid.values())]


# ---------------------------------------------------------------------------
# Reservoir generation dispatch (same logic as exp1_1)
# ---------------------------------------------------------------------------

_NEEDS_POOL = {
    "genomic",
    "gc_matched",
    "dinuc_shuffle",
    "prm_1pct",
    "prm_5pct",
    "prm_10pct",
    "prm_20pct",
    "prm_50pct",
    "prm_uniform_1_10",
    "recombination_uniform",
    "recombination_2pt",
    "evoaug_structural",
    "evoaug_heavy",
    "ise_maximize",
    "ise_diverse_targets",
    "ise_target_high",
    "snv",
}

_POOL_MUTAGENESIS = {
    "recombination_uniform",
    "recombination_2pt",
    "evoaug_structural",
    "evoaug_heavy",
}

_ISE_TYPES = {"ise_maximize", "ise_diverse_targets", "ise_target_high"}


def _generate_from_reservoir(
    reservoir_name: str,
    n: int,
    seed: int,
    task: str,
    pool_seqs: list[str] | None = None,
    pool_labels: np.ndarray | None = None,
    oracle: Any = None,
) -> list[str]:
    """Generate *n* sequences using the named reservoir."""
    res = _load_reservoir(reservoir_name, seed=seed)

    if reservoir_name == "random":
        seqs, _ = res.generate(n, task=task)
    elif reservoir_name == "dinuc_shuffle":
        seqs, _ = res.generate(n, task=task, method="dinuc_shuffle", reference_sequences=pool_seqs)
    elif reservoir_name == "genomic":
        seqs, _ = res.generate(n, pool_sequences=pool_seqs, pool_labels=pool_labels)
    elif reservoir_name == "gc_matched":
        seqs, _ = res.generate(n, pool_sequences=pool_seqs, task=task)
    elif reservoir_name.startswith("prm") or reservoir_name == "snv":
        seqs, _ = res.generate(n, base_sequences=pool_seqs, task=task)
    elif reservoir_name in _POOL_MUTAGENESIS or reservoir_name.startswith("recombination"):
        seqs, _ = res.generate(n, base_sequences=pool_seqs, task=task)
    elif reservoir_name in _ISE_TYPES:
        seqs, _ = res.generate(n, base_sequences=pool_seqs, task=task, student_model=oracle)
    else:
        seqs, _ = res.generate(n, task=task)

    return seqs


# ---------------------------------------------------------------------------
# Free GPU memory (shared utility)
# ---------------------------------------------------------------------------


def _free_gpu_memory() -> None:
    """Release GPU memory held by JAX / PyTorch."""
    gc.collect()
    try:
        import jax

        jax.clear_caches()
        backend = jax.lib.xla_bridge.get_backend()
        for buf in backend.live_buffers():
            buf.delete()
    except Exception:
        pass
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run_acquisition_experiment(
    task: str,
    student_type: str,
    reservoir_name: str,
    acquisition_name: str,
    oracle_type: str,
    regime: str,
    pool_ratio: int,
    n_replicates: int,
    seed: int,
    output_base: Path,
    hp_sweep: bool = True,
    ensemble_size: int = 5,
    epochs: int = 80,
    early_stop_patience: int | None = None,
) -> list[RunResult]:
    """Run one (reservoir x acquisition x regime) experiment.

    Returns a list of :class:`RunResult` for all HP configs and replicates.
    """
    from evaluation.exp1_eval import evaluate_on_exp1_test_panel

    task_cfg = TASK_CONFIGS[task]
    regime_cfg = DATA_REGIMES[regime]
    initial_size = regime_cfg["initial_size"]
    batch_size = regime_cfg["batch_size"]
    pool_size = pool_ratio * batch_size

    # Resolve test set directory
    resolved_oracle = (
        oracle_type if oracle_type != "default" else ("ag" if task == "k562" else "dream_rnn")
    )
    default_oracle = "ag" if task == "k562" else "dream_rnn"
    if resolved_oracle != default_oracle:
        test_set_dir = REPO / "data" / task / f"test_sets_{resolved_oracle}"
    else:
        test_set_dir = REPO / task_cfg["test_set_dir"]

    if not test_set_dir.exists():
        fallback_dir = REPO / task_cfg["test_set_dir"]
        if fallback_dir.exists():
            logger.warning(f"Test set dir {test_set_dir} not found. Falling back to {fallback_dir}")
            test_set_dir = fallback_dir

    # Output directory structure:
    # {output_base}/{reservoir}/{acquisition}/{regime}/hp{idx}/seed{N}/result.json
    exp_dir = output_base / reservoir_name / acquisition_name / regime

    # ── Phase 1: Generate data and cache oracle labels ─────────────────
    # Cache paths
    initial_cache = exp_dir / "initial_labels.npz"
    pool_cache = exp_dir / "pool_labels.npz"

    # Load genomic pool if needed by this reservoir
    pool_seqs, pool_labels = None, None
    if reservoir_name in _NEEDS_POOL:
        logger.info("Loading genomic pool sequences...")
        pool_seqs, pool_labels = _load_pool_sequences(task)
        logger.info(f"Pool size: {len(pool_seqs):,}")

    oracle = None

    if not initial_cache.exists() or not pool_cache.exists():
        logger.info(f"Loading oracle for task={task}, oracle_type={oracle_type}...")
        oracle = _load_oracle(task, oracle_type=oracle_type)

    # 1a. Initial labeled set (genomic subsample from training pool)
    if initial_cache.exists():
        logger.info(f"Loading cached initial set from {initial_cache}")
        cached = np.load(initial_cache, allow_pickle=True)
        initial_seqs = cached["sequences"].tolist()
        initial_labels = cached["labels"]
    else:
        logger.info(f"Generating initial set: {initial_size:,} genomic sequences")
        all_pool_seqs, _ = _load_pool_sequences(task)
        rng = np.random.default_rng(seed)
        initial_idx = rng.choice(len(all_pool_seqs), size=initial_size, replace=False)
        initial_seqs = [all_pool_seqs[i] for i in initial_idx]
        initial_labels = oracle.predict(initial_seqs)
        exp_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            initial_cache,
            sequences=np.array(initial_seqs, dtype=object),
            labels=initial_labels,
        )
        logger.info(f"Cached initial labels ({len(initial_seqs):,}) to {initial_cache}")
        del all_pool_seqs

    # 1b. Reservoir candidate pool
    if pool_cache.exists():
        logger.info(f"Loading cached pool from {pool_cache}")
        cached = np.load(pool_cache, allow_pickle=True)
        candidate_seqs = cached["sequences"].tolist()
        candidate_labels = cached["labels"]
    else:
        logger.info(f"Generating reservoir pool: {pool_size:,} sequences from '{reservoir_name}'")
        candidate_seqs = _generate_from_reservoir(
            reservoir_name,
            pool_size,
            seed=seed,
            task=task,
            pool_seqs=pool_seqs,
            pool_labels=pool_labels,
            oracle=oracle,
        )
        logger.info(f"Labeling {len(candidate_seqs):,} pool sequences with oracle...")
        label_start = time.perf_counter()
        candidate_labels = oracle.predict(candidate_seqs)
        logger.info(f"Oracle labeling took {time.perf_counter() - label_start:.1f}s")
        exp_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pool_cache,
            sequences=np.array(candidate_seqs, dtype=object),
            labels=candidate_labels,
        )
        logger.info(f"Cached pool labels ({len(candidate_seqs):,}) to {pool_cache}")

    # Free oracle from GPU before student training
    if oracle is not None:
        logger.info("Freeing oracle model from GPU memory...")
        del oracle
        oracle = None
        _free_gpu_memory()
        logger.info("Oracle freed.")

    # ── Phase 2: Per-replicate acquisition + retraining ────────────────
    total_n_train = initial_size + batch_size
    hp_configs = _build_hp_configs(student_type, total_n_train, hp_sweep)
    # Filter configs where batch_size > training samples
    hp_configs = [hp for hp in hp_configs if hp["batch_size"] <= total_n_train]
    logger.info(f"HP configs for total n={total_n_train:,}: {len(hp_configs)} configs")

    results: list[RunResult] = []

    for rep in range(n_replicates):
        rep_seed = seed + rep * 1000

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Replicate {rep + 1}/{n_replicates} (seed={rep_seed})")
        logger.info(f"{'=' * 60}")

        # 2a. Train initial student on the initial labeled set
        #     Use first HP config for the initial student (its quality matters
        #     for acquisition but we don't sweep it — the sweep is on the final
        #     retrained student).
        init_hp = hp_configs[0]
        logger.info(
            f"Training initial student ({student_type}) on {initial_size:,} seqs "
            f"(lr={init_hp['learning_rate']}, bs={init_hp['batch_size']})..."
        )

        # Pre-encode for AG S1
        initial_embs_cached = None
        if student_type in ("alphagenome_k562_s1", "alphagenome_yeast_s1"):
            ag = _get_ag_model_and_encoder(task)
            logger.info(f"  Pre-encoding {len(initial_seqs):,} initial seqs for AG S1...")
            initial_embs_cached = _encode_sequences_for_ag(initial_seqs, task, ag["encoder_fn"])

        init_student = _train_student(
            task=task,
            student_type=student_type,
            sequences=initial_seqs,
            labels=initial_labels,
            lr=init_hp["learning_rate"],
            batch_size=min(init_hp["batch_size"], len(initial_seqs)),
            seed=rep_seed,
            pre_encoded_embs=initial_embs_cached,
            ensemble_size=ensemble_size,
            epochs=epochs,
            early_stopping_patience=early_stop_patience,
        )

        # 2b. Run acquisition function
        logger.info(
            f"Running acquisition '{acquisition_name}' to select {batch_size:,} "
            f"from {len(candidate_seqs):,} pool candidates..."
        )
        acq_start = time.perf_counter()
        acq_fn = _make_acquisition(acquisition_name, seed=rep_seed)
        selected_idx = acq_fn.select(init_student, candidate_seqs, batch_size)
        acq_seconds = time.perf_counter() - acq_start
        logger.info(f"Acquisition took {acq_seconds:.1f}s")

        selected_seqs = [candidate_seqs[i] for i in selected_idx]
        selected_labels = candidate_labels[selected_idx]

        # 2c. Combine initial + selected
        combined_seqs = initial_seqs + selected_seqs
        combined_labels = np.concatenate([initial_labels, selected_labels])
        logger.info(f"Combined training set: {len(combined_seqs):,} sequences")

        # Free initial student
        del init_student

        # Validation split (10%)
        n_val = max(100, int(0.1 * len(combined_seqs)))
        rng = np.random.default_rng(rep_seed)
        val_idx = rng.choice(len(combined_seqs), size=n_val, replace=False)
        train_mask = np.ones(len(combined_seqs), dtype=bool)
        train_mask[val_idx] = False
        train_seqs = [combined_seqs[i] for i in range(len(combined_seqs)) if train_mask[i]]
        train_labels = combined_labels[train_mask]
        val_seqs = [combined_seqs[i] for i in val_idx]
        val_labels = combined_labels[val_idx]

        # Pre-encode for AG S1 (combined set)
        train_embs_cached = None
        if student_type in ("alphagenome_k562_s1", "alphagenome_yeast_s1"):
            ag = _get_ag_model_and_encoder(task)
            logger.info(f"  Pre-encoding {len(train_seqs):,} combined seqs for AG S1...")
            enc_start = time.perf_counter()
            train_embs_cached = _encode_sequences_for_ag(train_seqs, task, ag["encoder_fn"])
            logger.info(f"  Pre-encoding took {time.perf_counter() - enc_start:.1f}s")

        # 2d. HP sweep: retrain on combined data
        best_hp = None
        best_val_r = -1.0

        for hp_idx, hp in enumerate(hp_configs):
            run_dir = exp_dir / f"hp{hp_idx}" / f"seed{rep_seed}"

            # Skip completed runs
            result_path = run_dir / "result.json"
            if result_path.exists():
                logger.info(f"  Skipping completed run: {result_path}")
                try:
                    existing = json.loads(result_path.read_text())
                    results.append(
                        RunResult(**{k: existing[k] for k in RunResult.__dataclass_fields__})
                    )
                except Exception:
                    pass
                continue

            run_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"  HP {hp_idx + 1}/{len(hp_configs)}, "
                f"rep {rep + 1}/{n_replicates}, "
                f"lr={hp['learning_rate']}, bs={hp['batch_size']}"
            )

            run_start = time.perf_counter()
            try:
                student = _train_student(
                    task=task,
                    student_type=student_type,
                    sequences=train_seqs,
                    labels=train_labels,
                    lr=hp["learning_rate"],
                    batch_size=hp["batch_size"],
                    seed=rep_seed + hp_idx * 100,
                    pre_encoded_embs=train_embs_cached,
                    ensemble_size=ensemble_size,
                    epochs=epochs,
                    early_stopping_patience=early_stop_patience,
                )

                # Validation evaluation
                val_preds = student.predict(val_seqs)
                val_r = float(np.corrcoef(val_preds, val_labels)[0, 1])
                if np.isnan(val_r):
                    val_r = 0.0

                # Test evaluation
                test_metrics = evaluate_on_exp1_test_panel(student, task, test_set_dir)

                wall_s = time.perf_counter() - run_start

                result = RunResult(
                    reservoir=reservoir_name,
                    task=task,
                    student=student_type,
                    n_train=len(combined_seqs),
                    hp_config=hp,
                    seed=rep_seed,
                    val_pearson_r=val_r,
                    test_metrics=test_metrics,
                    wall_seconds=wall_s,
                    output_dir=str(run_dir),
                )
                results.append(result)

                # Save result with acquisition metadata
                result_dict = asdict(result)
                result_dict["acquisition"] = acquisition_name
                result_dict["regime"] = regime
                result_dict["initial_size"] = initial_size
                result_dict["batch_size"] = batch_size
                result_dict["pool_size"] = len(candidate_seqs)
                result_dict["acq_wall_seconds"] = acq_seconds
                (run_dir / "result.json").write_text(json.dumps(result_dict, indent=2, default=str))
                logger.info(f"    val_r={val_r:.4f}, wall={wall_s:.1f}s")

                # Track best HP
                if val_r > best_val_r:
                    best_val_r = val_r
                    best_hp = hp

                # W&B logging
                try:
                    import wandb

                    if wandb.run is not None:
                        log_data = {
                            "regime": regime,
                            "acquisition": acquisition_name,
                            "replicate": rep,
                            "val/pearson_r": val_r,
                            "hp/lr": hp["learning_rate"],
                            "hp/batch_size": hp["batch_size"],
                            "acq_wall_seconds": acq_seconds,
                        }
                        for tname, tmetrics in test_metrics.items():
                            for mname, mval in tmetrics.items():
                                log_data[f"test/{tname}/{mname}"] = mval
                        wandb.log(log_data)
                except ImportError:
                    pass

            except Exception as e:
                logger.error(f"    FAILED: {e}", exc_info=True)
                continue

        logger.info(f"  Best HP for rep {rep + 1}: {best_hp}, val_r={best_val_r:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1.2: Acquisition function benchmarking"
    )
    parser.add_argument("--task", required=True, choices=["k562", "yeast"])
    parser.add_argument(
        "--student",
        required=True,
        choices=[
            "dream_rnn",
            "alphagenome_k562_s1",
            "alphagenome_yeast_s1",
            "alphagenome_k562_s2",
            "alphagenome_yeast_s2",
        ],
    )
    parser.add_argument(
        "--reservoir",
        nargs="+",
        required=True,
        help="Reservoir strategy name(s) for generating the candidate pool",
    )
    parser.add_argument(
        "--acquisition",
        nargs="+",
        required=True,
        help="Acquisition function name(s): "
        + ", ".join(
            [
                "random",
                "uncertainty",
                "diversity",
                "badge",
                "combined",
                "ensemble",
                "prior_knowledge",
            ]
        ),
    )
    parser.add_argument(
        "--oracle",
        default="default",
        choices=["default", "ag", "dream_rnn"],
        help="Oracle type: 'default' (AG for K562, DREAM for yeast), 'ag', or 'dream_rnn'",
    )
    parser.add_argument(
        "--regime",
        nargs="+",
        default=["small"],
        choices=["small", "medium", "large"],
        help="Data regime(s) defining initial_size and batch_size",
    )
    parser.add_argument(
        "--pool-ratio",
        type=int,
        default=10,
        help="Pool size = pool_ratio * batch_size (default: 10)",
    )
    parser.add_argument("--n-replicates", type=int, default=3)
    parser.add_argument("--no-hp-sweep", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=5,
        help="Number of ensemble members per DREAM-RNN student (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Max training epochs for DREAM-RNN (default: 80)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs without improvement). Default: None",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    oracle_suffix = f"_{args.oracle}" if args.oracle != "default" else ""
    output_base = (
        Path(args.output_dir)
        if args.output_dir
        else (REPO / "outputs" / "exp1_2" / args.task / f"{args.student}{oracle_suffix}")
    )

    # W&B init
    try:
        import wandb

        wandb.init(
            project="albench-s2f",
            name=(
                f"exp1_2_{args.task}_{args.student}_"
                f"{'-'.join(args.acquisition)}_{'-'.join(args.regime)}"
            ),
            tags=["exp1", "acquisition", args.task, args.student] + args.acquisition + args.regime,
            group="exp1_2",
            config={
                "task": args.task,
                "student": args.student,
                "oracle": args.oracle,
                "reservoirs": args.reservoir,
                "acquisitions": args.acquisition,
                "regimes": args.regime,
                "pool_ratio": args.pool_ratio,
                "n_replicates": args.n_replicates,
                "hp_sweep": not args.no_hp_sweep,
            },
        )
    except (ImportError, Exception) as e:
        logger.info(f"wandb not available -- skipping logging ({e})")

    all_results: list[RunResult] = []

    for reservoir_name in args.reservoir:
        for acquisition_name in args.acquisition:
            for regime in args.regime:
                logger.info(f"\n{'#' * 70}")
                logger.info(
                    f"Reservoir: {reservoir_name} | "
                    f"Acquisition: {acquisition_name} | "
                    f"Regime: {regime}"
                )
                logger.info(f"{'#' * 70}")

                results = run_acquisition_experiment(
                    task=args.task,
                    student_type=args.student,
                    reservoir_name=reservoir_name,
                    acquisition_name=acquisition_name,
                    oracle_type=args.oracle,
                    regime=regime,
                    pool_ratio=args.pool_ratio,
                    n_replicates=args.n_replicates,
                    seed=args.seed,
                    output_base=output_base,
                    hp_sweep=not args.no_hp_sweep,
                    ensemble_size=args.ensemble_size,
                    epochs=args.epochs,
                    early_stop_patience=args.early_stop_patience,
                )
                all_results.extend(results)

    # Save summary
    summary_dir = output_base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = [asdict(r) for r in all_results]
    (summary_dir / "all_results.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"\nSaved {len(all_results)} results to {summary_dir}")

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
