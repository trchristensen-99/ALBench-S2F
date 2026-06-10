"""Ray Tune scheduler search engine for the HP bake-off (Step 1 of the redo).

Adds bandit-style HP search (ASHA, BOHB) as first-class "strategies" that sit
alongside the round-based suggest/update strategies in scaling_hp_search.py. Each
Ray *trial* is written out in the SAME on-disk contract as a round-based model —
`{model_id}.npz` (val/test/per-set predictions) + `{model_id}_meta.json` (hp,
val_pearson, train_time_sec, strategy) — so the downstream analysis (rounds curve,
GPU-seconds knees, all-subsets ElasticNet recipe) merges Ray trials into the same
pool transparently. strategy is labelled `ray_asha` / `ray_bohb`.

Why a separate engine: ASHA/BOHB own the trial-scheduling loop and decide
per-epoch early stopping; that does not fit the `suggest(n)->update()` interface.
The per-epoch hook they need already exists — LegNetStudent.fit(epoch_callback=)
forwards to train_model_optimized, which calls `epoch_callback(epoch, val_metrics)`
each epoch (val_metrics has 'pearson_r'). We report that to Ray and checkpoint on
improvement, so a scheduler-terminated trial still leaves a best checkpoint from
which the driver recovers predictions.

PBT is deliberately NOT implemented here: it requires a pause/perturb/resume
*step-wise* trainable (one epoch per step, reload-from-checkpoint with a mutated
config mid-training). The current LegNetStudent.fit() owns the whole epoch loop in
a single call, so PBT would need a separate per-epoch training driver — a larger
refactor. ASHA+BOHB deliver the core bandit value on the GPU-seconds efficiency
axis; add PBT incrementally if the bake-off shows it is worth the refactor.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

from experiments.scaling_hp_search import (
    LR_SCHEDULE_CHOICES,
    REPO,
    HPConfig,
    _atomic_savez,
    _atomic_write_text,
    build_block_sizes,
)

RAY_STRATEGIES = ("ray_asha", "ray_bohb")


# ── HP search space ───────────────────────────────────────────────────────────


def build_search_space():
    """Ray Tune search space over the core HPConfig axes.

    width_jitter (a per-layer, variable-length vector) is NOT a Ray dimension —
    it is drawn deterministically from `seed` inside the trainable so trials stay
    reproducible while Ray controls the scalar/categorical axes it can reason
    about. block_class/optimizer/ks/etc. mirror sample_random_hp's ranges.
    """
    from ray import tune

    return {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024]),
        "conv_dropout": tune.uniform(0.0, 0.3),
        "dense_dropout": tune.uniform(0.0, 0.5),
        "n_layers": tune.randint(2, 13),  # 2..12 (randint upper is exclusive)
        "width_base": tune.choice([16, 32, 64, 128, 256]),
        "block_class": tune.choice(["eff", "ag", "plain"]),
        "ks": tune.choice([3, 5, 7, 9, 11]),
        "pct_start": tune.choice([0.1, 0.2, 0.3, 0.4]),
        "optimizer": tune.choice(["adam", "adamw", "muon"]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "use_shift_aug": tune.choice([True, False]),
        "shift_max": tune.choice([5, 10, 15, 20]),
        "use_evoaug": tune.choice([True, False]),
        "lr_schedule": tune.choice(LR_SCHEDULE_CHOICES),
    }


def _patch_configspace_for_bohb():
    """Make Ray 2.54's BOHB adapter work with ConfigSpace>=1.0 without a venv
    downgrade.

    Ray's bohb_search.resolve_value() calls UniformFloat/IntegerHyperparameter
    with a `q=` (quantization) kwarg that ConfigSpace>=1.0 removed, so BOHB dies
    with `TypeError: __init__() got an unexpected keyword argument 'q'`. Our
    search space uses no quantization, so Ray always passes q=None here; wrap the
    two affected constructors to drop a None q. A non-None q would be silently
    lost, so refuse it loudly rather than corrupt the space.
    """
    import ConfigSpace as CS

    for name in ("UniformFloatHyperparameter", "UniformIntegerHyperparameter"):
        orig = getattr(CS, name, None)
        if orig is None or getattr(orig, "_bohb_q_patched", False):
            continue

        def _make(orig):
            def wrapper(*args, **kwargs):
                q = kwargs.pop("q", None)
                if q is not None:
                    raise ValueError(
                        f"ConfigSpace shim: non-None q={q} for {orig.__name__}; "
                        "the BOHB search space now uses quantization — update the shim."
                    )
                return orig(*args, **kwargs)

            wrapper._bohb_q_patched = True
            return wrapper

        setattr(CS, name, _make(orig))


def _hp_from_config(config: dict, seed: int) -> HPConfig:
    """Materialise a full HPConfig from a Ray trial config + a deterministic seed
    (used only for the width_jitter vector, so arch is reproducible)."""
    n_layers = int(config["n_layers"])
    rng = np.random.default_rng(seed)
    width_jitter = [float(2 ** rng.uniform(-1, 1)) for _ in range(n_layers)]
    return HPConfig(
        lr=float(config["lr"]),
        batch_size=int(config["batch_size"]),
        conv_dropout=float(config["conv_dropout"]),
        dense_dropout=float(config["dense_dropout"]),
        n_layers=n_layers,
        width_base=int(config["width_base"]),
        width_jitter=width_jitter,
        block_class=str(config["block_class"]),
        ks=int(config["ks"]),
        pct_start=float(config["pct_start"]),
        optimizer=str(config["optimizer"]),
        weight_decay=float(config["weight_decay"]),
        use_shift_aug=bool(config["use_shift_aug"]),
        shift_max=int(config["shift_max"]),
        use_evoaug=bool(config["use_evoaug"]),
        lr_schedule=str(config["lr_schedule"]),
        seed=seed,
    )


# ── Trainable ─────────────────────────────────────────────────────────────────


def _build_student(hp: HPConfig, epochs: int, device: str = "cuda"):
    """Build a single-member LegNetStudent for `hp` (mirrors train_one_model)."""
    from models.legnet_student import LegNetStudent, TrainConfig

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    block_sizes = build_block_sizes(hp.n_layers, hp.width_base, hp.width_jitter)
    train_cfg = TrainConfig(
        lr=hp.lr,
        batch_size=hp.batch_size,
        weight_decay=hp.weight_decay,
        epochs=epochs,
        pct_start=hp.pct_start,
        optimizer=hp.optimizer,
        evoaug_intensity="medium" if hp.use_evoaug else None,
        shift_aug=hp.use_shift_aug,
        max_shift=hp.shift_max,
        num_workers=4,
        use_compile=False,
        early_stopping_patience=epochs,  # let the scheduler own early stopping
    )
    student = LegNetStudent(
        task_mode="k562",
        ensemble_size=1,
        block_sizes=block_sizes,
        ks=hp.ks,
        block_class=hp.block_class,
        device=device,
        train_config=train_cfg,
        in_channels=4,
        conv_dropout=hp.conv_dropout,
        dense_dropout=hp.dense_dropout,
    )
    return student, block_sizes


def _trainable(config, *, data, epochs, seed):
    """Ray Tune function-API trainable. Trains one LegNet config, reporting
    val_pearson each epoch and checkpointing the model on improvement so a
    scheduler-terminated trial still leaves a recoverable best model."""
    import tempfile

    from ray import tune
    from ray.tune import Checkpoint

    hp = _hp_from_config(config, seed=seed)
    student, block_sizes = _build_student(hp, epochs=epochs)
    model = student.models[0]

    best = {"vp": float("-inf")}

    def epoch_callback(epoch, val_metrics):
        vp = float(val_metrics.get("pearson_r", float("nan")))
        if not np.isfinite(vp):
            tune.report({"val_pearson": -1.0, "epoch": epoch})
            return False
        improved = vp > best["vp"]
        if improved:
            best["vp"] = vp
            with tempfile.TemporaryDirectory() as ckpt_dir:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                (Path(ckpt_dir) / "hp.json").write_text(
                    json.dumps({"hp": asdict(hp), "block_sizes": block_sizes})
                )
                tune.report(
                    {"val_pearson": vp, "epoch": epoch},
                    checkpoint=Checkpoint.from_directory(ckpt_dir),
                )
        else:
            tune.report({"val_pearson": vp, "epoch": epoch})
        return False  # scheduler terminates externally; never self-stop

    student.fit(
        data["train_seqs"],
        data["train_labels"],
        val_sequences=data["val_seqs"],
        val_labels=data["val_labels"],
        epoch_callback=epoch_callback,
    )


# ── Driver ────────────────────────────────────────────────────────────────────


def _recover_and_write(result, model_id, strat_name, out_dir, data, extra_test_sets, round_idx):
    """Load a finished trial's best checkpoint, predict val/test/per-set, and
    write the {model_id}.npz + {model_id}_meta.json pair in run_search's format.

    round_idx makes each Ray trial its own point on the cumulative-GPU-seconds
    efficiency curve (aggregate_rounds_curve keys on meta['round'], one model per
    round), so the GPU-seconds knee analysis treats Ray identically to the
    round-based strategies."""
    from models.legnet_student import LegNetStudent, TrainConfig

    ckpt = result.checkpoint
    if ckpt is None:
        meta = {
            "model_id": model_id,
            "strategy": strat_name,
            "round": round_idx,
            "error": "no checkpoint (trial produced no improving epoch)",
            "hp": result.config,
        }
        _atomic_write_text(
            out_dir / f"{model_id}_meta.json", json.dumps(meta, indent=2, default=str)
        )
        return

    with ckpt.as_directory() as cdir:
        cdir = Path(cdir)
        saved = json.loads((cdir / "hp.json").read_text())
        hp = HPConfig(
            **{k: v for k, v in saved["hp"].items() if k in HPConfig.__dataclass_fields__}
        )
        block_sizes = saved["block_sizes"]
        train_cfg = TrainConfig(
            lr=hp.lr,
            batch_size=hp.batch_size,
            weight_decay=hp.weight_decay,
            epochs=1,
            pct_start=hp.pct_start,
            optimizer=hp.optimizer,
            num_workers=0,
            use_compile=False,
        )
        student = LegNetStudent(
            task_mode="k562",
            ensemble_size=1,
            block_sizes=block_sizes,
            ks=hp.ks,
            block_class=hp.block_class,
            device="cuda",
            train_config=train_cfg,
            in_channels=4,
            conv_dropout=hp.conv_dropout,
            dense_dropout=hp.dense_dropout,
        )
        student.models[0].load_state_dict(
            torch.load(cdir / "model.pt", map_location=student.device)
        )

    val_pred = student.predict(data["val_seqs"])
    test_pred = student.predict(data["test_seqs"])
    val_labels = data["val_labels"]
    val_r = float(pearsonr(val_pred, val_labels)[0])
    val_mse = float(((val_pred - val_labels) ** 2).mean())

    out = {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_pearson": val_r,
        "val_mse": val_mse,
        "train_time_sec": float(result.metrics.get("time_total_s", 0.0) or 0.0),
        "hp": asdict(hp),
        "block_sizes": block_sizes,
        "model_id": model_id,
        "strategy": strat_name,
        "round": round_idx,
        "n_epochs_trained": int(result.metrics.get("epoch", 0) or 0) + 1,
    }
    if extra_test_sets:
        per_set = {}
        for set_name, (seqs, oracle_labels) in extra_test_sets.items():
            pred = student.predict(seqs)
            out[f"test_pred_{set_name}"] = pred
            mask = np.isfinite(pred) & np.isfinite(oracle_labels)
            if mask.sum() >= 8:
                r = float(pearsonr(pred[mask], oracle_labels[mask])[0])
                mse = float(((pred[mask] - oracle_labels[mask]) ** 2).mean())
            else:
                r, mse = float("nan"), float("nan")
            per_set[set_name] = {"pearson": r, "mse": mse, "n": int(mask.sum())}
        out["per_set_metrics"] = per_set

    _atomic_savez(
        out_dir / f"{model_id}.npz", **{k: v for k, v in out.items() if isinstance(v, np.ndarray)}
    )
    meta = {k: v for k, v in out.items() if not isinstance(v, np.ndarray)}
    _atomic_write_text(out_dir / f"{model_id}_meta.json", json.dumps(meta, indent=2, default=str))


def run_ray_tune_search(args, scheduler_name: str):
    """Run a Ray Tune bake-off for one scheduler (ray_asha | ray_bohb).

    num_samples = args.rounds * args.per_strategy_per_round (match the round-based
    engine's total trial budget so the GPU-seconds efficiency comparison is fair).
    """
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    from experiments.scaling_hp_search import (
        load_all_test_sets,
        load_chr_test_genomic,
        load_chr_train_pool,
    )

    if scheduler_name not in RAY_STRATEGIES:
        raise ValueError(f"Unknown Ray scheduler: {scheduler_name}. Available: {RAY_STRATEGIES}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Ray Tune [{scheduler_name}] loading data (D={args.D}) ===", flush=True)
    train_seqs, train_labels, val_seqs, val_labels = load_chr_train_pool(
        args.D,
        ref_only=args.ref_only,
        val_frac=0.1,
        seed=args.data_seed,
        reservoir_cache=getattr(args, "reservoir_cache", None),
        chr_val=getattr(args, "chr_val", False),
    )
    test_seqs, test_oracle, test_true = load_chr_test_genomic()
    extra_test_sets = load_all_test_sets()

    # Save labels once (same contract as run_search).
    label_dict = {"val_labels": val_labels, "test_oracle": test_oracle, "test_true": test_true}
    for set_name, (_, oracle_labels) in extra_test_sets.items():
        label_dict[f"oracle_{set_name}"] = oracle_labels
    _atomic_savez(out_dir / "labels.npz", **label_dict)

    data = {
        "train_seqs": train_seqs,
        "train_labels": train_labels,
        "val_seqs": val_seqs,
        "val_labels": val_labels,
        "test_seqs": test_seqs,
    }

    num_samples = max(1, args.rounds * args.per_strategy_per_round)
    grace = max(1, args.epochs // 4)

    if scheduler_name == "ray_asha":
        scheduler = ASHAScheduler(
            metric="val_pearson",
            mode="max",
            max_t=args.epochs,
            grace_period=grace,
            reduction_factor=3,
        )
        search_alg = None
        param_space = build_search_space()
    elif scheduler_name == "ray_bohb":
        from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
        from ray.tune.search.bohb import TuneBOHB

        _patch_configspace_for_bohb()
        scheduler = HyperBandForBOHB(
            metric="val_pearson",
            mode="max",
            max_t=args.epochs,
            reduction_factor=3,
        )
        search_alg = TuneBOHB(metric="val_pearson", mode="max", seed=args.hp_seed)
        param_space = build_search_space()

    # Ray walks/packages the driver CWD as the runtime-env working_dir even for a
    # local instance. The repo root is ~470 GB (outputs/ + logs/), so packaging it
    # stalls indefinitely. Point working_dir at a tiny empty dir; trial workers
    # import via the absolute PYTHONPATH the job sets and receive data through the
    # object store (tune.with_parameters), so no repo files need shipping.
    ray_wd = out_dir / "_ray_wd"
    ray_wd.mkdir(parents=True, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
        runtime_env={"working_dir": str(ray_wd)},
    )

    trainable = tune.with_parameters(
        _trainable,
        data=data,
        epochs=args.epochs,
        seed=args.hp_seed,
    )
    trainable = tune.with_resources(trainable, {"gpu": 1, "cpu": 4})

    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
        ),
        # tune.RunConfig (verbose default None), NOT train.RunConfig whose verbose
        # defaults to the string "DEPRECATED" and crashes get_air_verbosity().
        run_config=tune.RunConfig(
            name=f"{scheduler_name}_D{args.D}_s{args.hp_seed}",
            storage_path=str((out_dir / "ray_results").resolve()),
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val_pearson",
                checkpoint_score_order="max",
            ),
        ),
    )
    results = tuner.fit()

    print(f"=== Ray Tune [{scheduler_name}] writing {len(results)} trials ===", flush=True)
    for i, result in enumerate(results):
        # r{i:02d}_ prefix → each trial is a distinct "round" so the rounds/GPU-sec
        # analysis (globs r*_meta.json, keys on meta['round']) treats it as a point.
        model_id = f"r{i:02d}_{scheduler_name}_00"
        if (out_dir / f"{model_id}_meta.json").exists():
            continue
        try:
            _recover_and_write(result, model_id, scheduler_name, out_dir, data, extra_test_sets, i)
            print(
                f"  wrote {model_id} (val_pearson={result.metrics.get('val_pearson')})", flush=True
            )
        except Exception as e:
            print(f"  ERROR writing {model_id}: {e}", flush=True)

    summary = {
        "strategy": scheduler_name,
        "num_samples": num_samples,
        "D": args.D,
        "ref_only": args.ref_only,
        "epochs": args.epochs,
        "grace_period": grace,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    ray.shutdown()
    print(
        f"=== Ray Tune [{scheduler_name}] done. {len(results)} trials in {out_dir} ===", flush=True
    )
