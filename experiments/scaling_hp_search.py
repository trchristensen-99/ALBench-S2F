"""HP search v2 — chr-split + cached data + 10% holdout val + optional compile + TF32.

Key changes from v1:
  - Uses pre-built chr_split_cache (faster startup, no DATA-Table_S2 reload)
  - 10% random holdout from train pool serves as val (oracle-labeled, consistent
    with other reservoir strategy protocols)
  - Disables torch.compile via TrainConfig.use_compile=False to skip autotune
  - Sets torch.set_float32_matmul_precision('high') for TF32 on H100/A100
  - Eval target = chr-test genomic with ORACLE labels (primary)
    + report real labels too as secondary metric
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

from experiments.test_set_guards import assert_mono_snv

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
CACHE = REPO / "outputs/chr_split_cache"

# TF32 — fast on H100 with negligible accuracy loss
torch.set_float32_matmul_precision("high")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (tmp file + os.replace). A resumed job — or any
    reader — never sees a half-written checkpoint if this process is killed
    mid-write (walltime SIGKILL, preemption, node crash). os.replace is atomic
    on POSIX, so the target is always either the old file or the complete new one."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays) -> None:
    """np.savez to a tmp .npz then atomic-replace (see _atomic_write_text)."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ── Data loading (cached) ─────────────────────────────────────────────────────


def load_chr_train_pool(
    D: int | None,
    ref_only: bool = True,
    val_frac: float = 0.1,
    seed: int = 0,
    reservoir_cache: str | Path | None = None,
    chr_val: bool = False,
):
    """Load chr-train pool with oracle labels; carve out val.

    Returns (train_seqs, train_labels, val_seqs, val_labels) — all oracle labels.

    When chr_val=False (default), val = 10% random holdout from the training pool.
    When chr_val=True, val = real chr19+21+X genomic sequences from
    outputs/chr_split_cache/chr_val_ref_only.npz (the "production" val protocol).
    For genomic-based strategies this matches what's used at inference time.
    For synthetic strategies (e.g. random, motif_planted) chr_val still works —
    train comes from the reservoir cache, val from real chr-val.
    """
    if reservoir_cache is not None:
        cache_path = Path(reservoir_cache)
        z = np.load(cache_path, allow_pickle=True)
        fname = cache_path.name
    else:
        fname = "chr_train_ref_only.npz" if ref_only else "chr_train_all_alleles.npz"
        z = np.load(CACHE / fname, allow_pickle=True)
    seqs = z["sequences"]
    labels = z["oracle_labels"]
    n = len(seqs)
    print(f"  Loaded {fname}: n={n:,}, μ={labels.mean():.3f} σ={labels.std():.3f}")

    if D is not None and D < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=D, replace=False)
        seqs = seqs[idx]
        labels = labels[idx]
        print(f"  Subsampled to D={D:,}")
    elif D is not None and D > n:
        print(f"  WARN: D={D:,} > pool size {n:,}; using all available")

    if chr_val:
        # Real chr19+21+X val with oracle labels — production-style holdout.
        cv = np.load(CACHE / "chr_val_ref_only.npz", allow_pickle=True)
        val_seqs = [str(s) for s in cv["sequences"]]
        val_labels = cv["oracle_labels"].astype(np.float32)
        # Filter to finite labels (some real sequences have NaN oracle on bad inputs)
        finite = np.isfinite(val_labels)
        if not finite.all():
            val_seqs = [s for s, ok in zip(val_seqs, finite) if ok]
            val_labels = val_labels[finite]
        train_seqs = [str(s) for s in seqs]
        train_labels = labels.astype(np.float32)
        print(f"  Train={len(train_seqs):,}  Val(chr19+21+X)={len(val_seqs):,}")
    else:
        # 10% holdout for val (legacy / synthetic-strategy path)
        rng2 = np.random.default_rng(seed + 1)
        perm = rng2.permutation(len(seqs))
        n_val = int(val_frac * len(seqs))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        val_seqs = [str(s) for s in seqs[val_idx]]
        val_labels = labels[val_idx].astype(np.float32)
        train_seqs = [str(s) for s in seqs[train_idx]]
        train_labels = labels[train_idx].astype(np.float32)
        print(f"  Train={len(train_seqs):,}  Val(10% holdout)={len(val_seqs):,}")
    return train_seqs, train_labels, val_seqs, val_labels


def load_chr_test_genomic():
    """Chr-test genomic ref-only sequences with oracle (eval target) + real labels (secondary)."""
    z = np.load(REPO / "data/k562/test_sets_ag_s2_chrsplit/genomic_oracle.npz", allow_pickle=True)
    seqs = [str(s) for s in z["sequences"]]
    oracle = z["oracle_mean"].astype(np.float32)
    true_label = z["true_label"].astype(np.float32)
    print(f"  Chr-test genomic: n={len(seqs):,}")
    return seqs, oracle, true_label


# Comprehensive test-set battery (all under data/k562/test_sets_ag_s2_chrsplit/)
# Each *_oracle.npz has keys: sequences (object), oracle_mean (float32).
# Generated by scripts/build_comprehensive_test_sets.py (one-time).
TEST_SET_NAMES = [
    "genomic",  # 32k chr 7+13 (canonical in-dist; loaded above too)
    "ood",  # 22k designed sequences (out-of-distribution)
    "snv",  # SNV ref/alt pairs for SNV effect prediction
    "random_32k",  # 32k uniformly random
    "dinuc_shuffle",  # 32k dinuc-preserving shuffle of genomic
    "sub_low",
    "sub_med",
    "sub_high",
    "ins_low",
    "ins_med",
    "ins_high",
    "del_low",
    "del_med",
    "del_high",
    "translocation",
    "inversion",
]


def load_all_test_sets() -> dict:
    """Return {set_name: (sequences, oracle_labels)} for every test set that exists on disk.

    Missing or non-standard-schema sets are silently skipped so scaling_hp_search.py
    works even before the full battery is materialized (e.g. during incremental
    rollout). SNV has a paired ref/alt schema — handled separately as snv_ref and
    snv_alt sets if the file is present.
    """
    test_dir = REPO / "data/k562/test_sets_ag_s2_chrsplit"
    # Eager, loud SNV guard: a present-but-wrong SNV file must halt the run, not be
    # silently skipped by the tolerant per-set loop below (which only tolerates a
    # *missing* set during incremental rollout).
    snv_path = test_dir / "snv_oracle.npz"
    if snv_path.exists():
        assert_mono_snv(np.load(snv_path, allow_pickle=True), snv_path)
    out = {}
    for name in TEST_SET_NAMES:
        path = test_dir / f"{name}_oracle.npz"
        if not path.exists():
            # legacy "random_10k" fallback
            if name == "random_32k" and (test_dir / "random_10k_oracle.npz").exists():
                path = test_dir / "random_10k_oracle.npz"
            else:
                continue
        try:
            z = np.load(path, allow_pickle=True)
            if name == "snv":
                # paired ref/alt schema → split into 2 sets
                if "ref_sequences" not in z.files:
                    continue
                ref_seqs = [str(s) for s in z["ref_sequences"]]
                alt_seqs = [str(s) for s in z["alt_sequences"]]
                ref_lab = z["ref_mean"].astype(np.float32)
                alt_lab = z["alt_mean"].astype(np.float32)
                out["snv_ref"] = (ref_seqs, ref_lab)
                out["snv_alt"] = (alt_seqs, alt_lab)
                print(
                    f"  test set snv_ref         : n={len(ref_seqs):,}  μ={ref_lab.mean():+.3f} σ={ref_lab.std():.3f}"
                )
                print(
                    f"  test set snv_alt         : n={len(alt_seqs):,}  μ={alt_lab.mean():+.3f} σ={alt_lab.std():.3f}"
                )
                continue
            # Standard schema: sequences + oracle_mean (or oracle_labels)
            if "sequences" not in z.files:
                print(f"  [skip] {name}: no 'sequences' key (keys={z.files})")
                continue
            seqs = [str(s) for s in z["sequences"]]
            if "oracle_mean" in z.files:
                labels = z["oracle_mean"]
            elif "oracle_labels" in z.files:
                labels = z["oracle_labels"]
            else:
                print(f"  [skip] {name}: no oracle_mean/oracle_labels key")
                continue
            labels = labels.astype(np.float32)
            out[name] = (seqs, labels)
            print(
                f"  test set {name:<16s}: n={len(seqs):,}  μ={labels.mean():+.3f} σ={labels.std():.3f}"
            )
        except Exception as e:
            print(f"  [skip] {name}: error loading — {type(e).__name__}: {str(e)[:100]}")
    return out


# ── HP space ──────────────────────────────────────────────────────────────────


@dataclass
class HPConfig:
    lr: float
    batch_size: int
    conv_dropout: float
    dense_dropout: float
    n_layers: int
    width_base: int
    width_jitter: list  # per-layer multiplier in [0.5, 2.0]
    block_class: str  # {"eff", "local"}
    ks: int  # kernel size
    pct_start: float  # OneCycleLR warmup fraction
    optimizer: str
    weight_decay: float
    use_shift_aug: bool
    shift_max: int
    use_evoaug: bool
    seed: int


def sample_random_hp(rng: np.random.Generator, seed: int) -> HPConfig:
    n_layers = int(rng.integers(2, 13))  # 2 to 12
    width_jitter = [float(2 ** rng.uniform(-1, 1)) for _ in range(n_layers)]
    return HPConfig(
        lr=float(10 ** rng.uniform(-5, -2)),
        batch_size=int(rng.choice([32, 64, 128, 256, 512, 1024])),
        conv_dropout=float(rng.uniform(0, 0.3)),
        dense_dropout=float(rng.uniform(0, 0.5)),
        n_layers=n_layers,
        width_base=int(rng.choice([16, 32, 64, 128, 256])),
        width_jitter=width_jitter,
        block_class=str(rng.choice(["eff", "ag", "plain"])),
        ks=int(rng.choice([3, 5, 7, 9, 11])),
        pct_start=float(rng.choice([0.1, 0.2, 0.3, 0.4])),
        optimizer=str(rng.choice(["adam", "adamw", "muon"])),
        weight_decay=float(10 ** rng.uniform(-6, -2)),
        use_shift_aug=bool(rng.random() < 0.5),
        shift_max=int(rng.choice([5, 10, 15, 20])),
        use_evoaug=bool(rng.random() < 0.3),
        seed=seed,
    )


def build_block_sizes(
    n_layers: int, width_base: int, width_jitter: list | None = None
) -> list[int]:
    """Per-layer widths with jitter; caps max to 2*width_base to prevent blow-up."""
    if width_jitter is None:
        width_jitter = [1.0] * n_layers
    sizes = []
    for i in range(n_layers):
        # Pattern: layers 0-1 = width_base, 2-3 = /2, 4-5 = /4, 6+ = /8 (floor 8)
        tier = i // 2
        mult = max(0.125, 2.0 ** (-tier))  # 1, 1, 0.5, 0.5, 0.25, 0.25, ...
        jitter = width_jitter[i] if i < len(width_jitter) else 1.0
        sz = max(8, int(round(width_base * mult * jitter / 8)) * 8)
        sz = min(sz, 2 * width_base)  # cap at 2x base for jitter safety
        sizes.append(sz)
    return sizes


# ── Training ──────────────────────────────────────────────────────────────────


def train_one_model(
    hp: HPConfig,
    train_seqs,
    train_labels,
    val_seqs,
    val_labels,
    test_seqs,
    epochs: int = 30,
    device: str = "cuda",
    use_compile: bool = False,
    early_stopping_patience: int = 8,
    extra_test_sets: dict | None = None,
):
    import sys

    sys.path.insert(0, str(REPO))
    from models.legnet_student import LegNetStudent, TrainConfig

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    width_jitter = hp.width_jitter if hp.width_jitter else [1.0] * hp.n_layers
    block_sizes = build_block_sizes(hp.n_layers, hp.width_base, width_jitter)
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
        use_compile=use_compile,
        early_stopping_patience=early_stopping_patience,
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
    t0 = time.time()
    student.fit(train_seqs, train_labels, val_sequences=val_seqs, val_labels=val_labels)
    train_time = time.time() - t0

    val_pred = student.predict(val_seqs)
    test_pred = student.predict(test_seqs)
    val_r = float(pearsonr(val_pred, val_labels)[0])
    val_mse = float(((val_pred - val_labels) ** 2).mean())

    # Epoch diagnostics from the (single-member) training history
    epoch_diag = {}
    if student.histories:
        h = student.histories[0]
        vp = h.get("val_pearson_r", [])
        if vp:
            best_ep = int(np.argmax(vp))
            epoch_diag = {
                "best_epoch": best_ep,
                "epochs_trained": len(vp),
                "early_stopped": len(vp) < epochs,
                "best_val_pearson": float(vp[best_ep]),
            }

    result = {
        "val_pred": val_pred,
        "test_pred": test_pred,  # backward-compat: this is the genomic test set
        "val_pearson": val_r,
        "val_mse": val_mse,
        "train_time_sec": train_time,
        "hp": asdict(hp),
        "block_sizes": block_sizes,
        **epoch_diag,
    }

    # Predict on all extra test sets (comprehensive eval battery)
    if extra_test_sets:
        per_set_metrics = {}
        for set_name, (seqs, oracle_labels) in extra_test_sets.items():
            pred = student.predict(seqs)
            result[f"test_pred_{set_name}"] = pred
            # Pairwise metrics vs oracle labels (skip NaN-safe just in case)
            mask = np.isfinite(pred) & np.isfinite(oracle_labels)
            if mask.sum() >= 8:
                r = float(pearsonr(pred[mask], oracle_labels[mask])[0])
                mse = float(((pred[mask] - oracle_labels[mask]) ** 2).mean())
            else:
                r, mse = float("nan"), float("nan")
            per_set_metrics[set_name] = {"pearson": r, "mse": mse, "n": int(mask.sum())}
        result["per_set_metrics"] = per_set_metrics

    return result


# ── Multi-strategy driver ─────────────────────────────────────────────────────


def _preload_history(out_dir: Path, strategies: dict) -> int:
    """Seed each strategy's history from already-saved *_meta.json so a resumed
    run's suggest() calls see all prior results. Returns count of records loaded."""
    n = 0
    per_strat: dict[str, tuple[list, list]] = {name: ([], []) for name in strategies}
    for meta_path in sorted(out_dir.glob("r*_meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        name = meta.get("strategy")
        if name not in per_strat or "val_pearson" not in meta:
            continue
        hp_fields = {
            k: v for k, v in meta.get("hp", {}).items() if k in HPConfig.__dataclass_fields__
        }
        per_strat[name][0].append(HPConfig(**hp_fields))
        per_strat[name][1].append(meta["val_pearson"])
        n += 1
    for name, (cs, vs) in per_strat.items():
        if cs:
            strategies[name].update(cs, vs)
            print(f"  [resume] preloaded {len(cs)} prior results into '{name}' history")
    return n


def run_search(args):
    import sys

    sys.path.insert(0, str(REPO))
    from experiments.hp_strategies import get_strategy

    try:
        from experiments import llm_autoresearch  # registers llm_autoresearch strategy
    except ImportError:
        pass  # anthropic SDK not available
    try:
        from experiments.llm_autoresearch import RateLimitExceeded
    except Exception:

        class RateLimitExceeded(Exception):  # fallback: never raised w/o LLM strategy
            pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Loading data (D={args.D}, ref_only={args.ref_only}, reservoir_cache={getattr(args, 'reservoir_cache', None)}, chr_val={getattr(args, 'chr_val', False)}) ==="
    )
    train_seqs, train_labels, val_seqs, val_labels = load_chr_train_pool(
        args.D,
        ref_only=args.ref_only,
        val_frac=0.1,
        seed=args.data_seed,
        reservoir_cache=getattr(args, "reservoir_cache", None),
        chr_val=getattr(args, "chr_val", False),
    )
    test_seqs, test_oracle, test_true = load_chr_test_genomic()

    # Load comprehensive eval battery (may be partial; missing files are skipped).
    print("=== Loading comprehensive test-set battery ===")
    all_test_sets = load_all_test_sets()
    # The "genomic" entry in all_test_sets duplicates load_chr_test_genomic — that's ok;
    # the analysis pipeline expects per-set predictions for all sets, including genomic.

    # Save labels once: legacy keys + per-set oracle labels for downstream analysis
    label_dict = {
        "val_labels": val_labels,
        "test_oracle": test_oracle,
        "test_true": test_true,
    }
    for set_name, (_, oracle_labels) in all_test_sets.items():
        label_dict[f"oracle_{set_name}"] = oracle_labels
    _atomic_savez(out_dir / "labels.npz", **label_dict)

    # Build strategies
    strategy_names = args.strategies.split(",")
    strategies = {
        name: get_strategy(name, seed=args.hp_seed + i * 1000)
        for i, name in enumerate(strategy_names)
    }
    print(f"=== Strategies: {list(strategies)} ===")

    # Resume: seed strategy history from any results already on disk.
    n_preloaded = _preload_history(out_dir, strategies)
    if n_preloaded:
        print(f"=== Resume: preloaded {n_preloaded} prior model results ===")

    total = 0
    for rd in range(args.rounds):
        print(f"\n=== Round {rd + 1}/{args.rounds} ===")
        # Proposal checkpoint: persist this round's configs so a resumed run reuses
        # the SAME configs (model_ids stay stable) instead of re-calling the LLM.
        proposals_path = out_dir / f"round_{rd:02d}_proposals.json"
        saved = None
        if proposals_path.exists():
            try:
                saved = json.loads(proposals_path.read_text())
            except Exception as e:
                # Corrupt/partial checkpoint (e.g. killed mid-write before atomic
                # writes landed). Don't crash-loop — discard and re-propose below.
                print(f"  [resume] proposals for round {rd} unreadable ({e}); re-proposing")
                saved = None
        if saved is not None:
            round_configs = [
                (
                    item["strategy"],
                    HPConfig(
                        **{
                            k: v
                            for k, v in item["hp"].items()
                            if k in HPConfig.__dataclass_fields__
                        }
                    ),
                )
                for item in saved
            ]
            print(f"  [resume] loaded {len(round_configs)} saved proposals for round {rd}")
        else:
            round_configs = []
            try:
                for name, strat in strategies.items():
                    n_per_strat = args.per_strategy_per_round
                    configs = strat.suggest(n_per_strat)
                    for c in configs:
                        round_configs.append((name, c))
                    print(f"  {name}: proposed {len(configs)} configs")
            except RateLimitExceeded as e:
                # FAIRNESS: never fall back to random. Checkpoint (completed models
                # are already on disk) and stop cleanly so this job can be resumed.
                print(f"\n=== RATE-LIMITED during proposal (round {rd}): {e} ===", flush=True)
                print(
                    f"=== Stopping cleanly. Re-launch the SAME command to resume "
                    f"(rounds 0..{rd - 1} done, history will be preloaded). ===",
                    flush=True,
                )
                sys.exit(42)
            _atomic_write_text(
                proposals_path,
                json.dumps(
                    [{"strategy": n_, "hp": asdict(c)} for n_, c in round_configs], indent=2
                ),
            )

        # Train each config sequentially (single-GPU for now)
        # Track only models trained in THIS run; preloaded history already covers
        # results on disk, so re-adding them here would double-count.
        newly_trained: dict[str, tuple[list, list]] = {name: ([], []) for name in strategies}
        for i, (strat_name, hp) in enumerate(round_configs):
            model_id = f"r{rd:02d}_{strat_name}_{i:02d}"
            # Skip already-completed models (resume): meta file present = attempted.
            if (out_dir / f"{model_id}_meta.json").exists():
                print(f"  [resume] skip {model_id} (already done)")
                total += 1
                continue
            print(
                f"\n  Training {model_id}: lr={hp.lr:.1e} bs={hp.batch_size} "
                f"layers={hp.n_layers} width={hp.width_base} opt={hp.optimizer}"
            )
            try:
                esp = getattr(args, "early_stop_patience", None) or 10
                result = train_one_model(
                    hp,
                    train_seqs,
                    train_labels,
                    val_seqs,
                    val_labels,
                    test_seqs,
                    epochs=args.epochs,
                    use_compile=False,
                    early_stopping_patience=esp,
                    extra_test_sets=all_test_sets,
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                result = {"hp": asdict(hp), "error": str(e), "strategy": strat_name, "round": rd}
            result["model_id"] = model_id
            result["strategy"] = strat_name
            result["round"] = rd
            _atomic_savez(
                out_dir / f"{model_id}.npz",
                **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
            )
            meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
            # meta written LAST + atomically: it is the resume gate (its presence
            # means "model done"), so it must never exist in a partial state and
            # must not appear before its companion .npz is fully on disk.
            _atomic_write_text(
                out_dir / f"{model_id}_meta.json", json.dumps(meta, indent=2, default=str)
            )
            # Log to global KB
            if result.get("val_mse") is not None:
                try:
                    from experiments.hp_knowledge_base import get_kb

                    get_kb().add(
                        hp=result.get("hp", {}),
                        val_metric=result.get("val_mse"),
                        context={
                            "D": args.D,
                            "ref_only": args.ref_only,
                            "epochs": args.epochs,
                            "strategy": strat_name,
                            "round": rd,
                        },
                    )
                except Exception as e:
                    print(f"    KB log failed: {e}", flush=True)
            if "val_pearson" in result:
                newly_trained[strat_name][0].append(hp)
                newly_trained[strat_name][1].append(result["val_pearson"])
            total += 1
            print(
                f"    val_pearson={result.get('val_pearson', 'ERR')}  time={result.get('train_time_sec', 0):.1f}s"
            )

        # Update each strategy with ONLY this run's newly-trained results
        # (preloaded history already accounts for results from prior runs).
        for name, strat in strategies.items():
            cs, vs = newly_trained[name]
            if cs:
                strat.update(cs, vs)

    # Final summary
    summary = {
        "strategies": list(strategies),
        "rounds": args.rounds,
        "per_strategy_per_round": args.per_strategy_per_round,
        "total_models": total,
        "D": args.D,
        "ref_only": args.ref_only,
        "epochs": args.epochs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Done. {total} models trained. Run aggregate_ensemble.py on {out_dir} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategies",
        default="random",
        help="comma list, e.g. random,optuna_tpe,autoresearch_single",
    )
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--per_strategy_per_round", type=int, default=5)
    ap.add_argument("--D", type=int, default=10_000)
    ap.add_argument("--ref_only", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hp_seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument(
        "--reservoir_cache",
        type=str,
        default=None,
        help="Path to a pre-generated reservoir-cache npz (sequences, oracle_labels) "
        "to use instead of chr_train_ref_only.npz. Used for non-genomic "
        "reservoir strategies (random, prm_*, evoaug_*, motif_planted, etc.). "
        "Generate with scripts/generate_reservoir_cache.py.",
    )
    ap.add_argument(
        "--chr_val",
        action="store_true",
        help="Use real chr19+21+X as the validation set (loads from chr_val_ref_only.npz) "
        "instead of a 10%% random holdout from train. Use this for genomic-based "
        "strategies so val matches the production protocol.",
    )
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="Override early stopping patience (default 10). "
        "Use lower (e.g. 5) for fair-budget fixed-cost scaling runs.",
    )
    args = ap.parse_args()

    # Ray Tune scheduler engines (ray_asha/ray_bohb) own the whole trial loop, so
    # they cannot share a process with the round-based suggest/update strategies.
    # Require a Ray strategy to be the sole strategy for its out_dir.
    from experiments.ray_tune_search import RAY_STRATEGIES, run_ray_tune_search

    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    ray_requested = [s for s in requested if s in RAY_STRATEGIES]
    if ray_requested:
        if len(requested) > 1:
            raise SystemExit(
                f"Ray scheduler strategies own the search loop and must be run alone "
                f"(one per out_dir). Got --strategies={args.strategies}."
            )
        run_ray_tune_search(args, ray_requested[0])
        return

    run_search(args)


if __name__ == "__main__":
    main()
