"""Multi-reservoir HP search driver.

Same K strategies, same per_round, but each proposed HP config is trained on
ALL --cross_R_caches simultaneously, val_pearsons are averaged across reservoirs,
and the mean is fed back to the strategy. This produces HP configs that are
robust by construction (the strategy never sees a single-reservoir val signal).

Output structure (compatible with existing analysis tools):
  {out_root}/{R}/{strat}/r{rd:02d}_{strat}_{i:02d}.npz + _meta.json

Compare with regular per-cell pilot: same models pooled across reservoirs vs
cross-R driven configs.

Usage:
  uv run --no-sync python experiments/scaling_hp_search_multi.py \\
    --strategies optuna_gp,evo_batch,llm_explore_nv1,evo_single,optuna_tpe \\
    --rounds 50 --per_strategy_per_round 1 \\
    --D 30000 --ref_only \\
    --cross_R "genomic=,motif_planted_v2=outputs/reservoir_cache/k562_motif_planted_v2_d1000000_seed42.npz,dinuc_shuffle=outputs/reservoir_cache/k562_dinuc_shuffle_d1000000_seed42.npz" \\
    --cross_R_val "genomic=chr_val,motif_planted_v2=outputs/reservoir_val_cache/k562_motif_planted_v2_val_seed42.npz,dinuc_shuffle=" \\
    --data_seed 42 --hp_seed 0 \\
    --epochs 100 --early_stop_patience 15 --min_delta 1e-3 \\
    --out_dir outputs/hp_multi_d30000/seed42_0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, str(REPO))

from experiments.scaling_hp_search import (  # noqa: E402
    HPConfig,
    _atomic_savez,
    _atomic_write_text,
    _hp_to_dict,
    build_regime,
    load_all_test_sets,
    load_battery_provenance,
    load_chr_test_genomic,
    load_chr_train_pool,
    regime_key,
    train_one_model,
)


def parse_R_spec(spec: str) -> list[tuple[str, str | None]]:
    """Parse 'R1=path1,R2=path2,R3=' into [(R1, path1), (R2, path2), (R3, None)].
    Empty value means: no reservoir cache (genomic uses chr_train_ref_only)
    or no reservoir_val cache (falls back to per-combo 10% holdout)."""
    out = []
    for tok in spec.split(","):
        if "=" not in tok:
            raise ValueError(f"bad spec token (need 'R=path'): {tok}")
        R, p = tok.split("=", 1)
        out.append((R.strip(), p.strip() or None))
    return out


def load_R_data(
    D: int, ref_only: bool, data_seed: int, reservoir: str, cache: str | None, val_spec: str | None
):
    """Load one reservoir's train/val pool. val_spec: None|"chr_val"|<path>."""
    if val_spec == "chr_val":
        chr_val_flag = True
        val_cache = None
    elif val_spec:
        chr_val_flag = False
        val_cache = val_spec
    else:
        chr_val_flag = False
        val_cache = None
    return load_chr_train_pool(
        D,
        ref_only=ref_only,
        val_frac=0.1,
        seed=data_seed,
        reservoir_cache=cache,
        chr_val=chr_val_flag,
        reservoir_val_cache=val_cache,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--per_strategy_per_round", type=int, default=1)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--ref_only", action="store_true")
    ap.add_argument("--out_dir", required=True, help="Root dir; per-R subdirs will be created.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hp_seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    ap.add_argument(
        "--cross_R",
        required=True,
        help="Comma list of 'R=cache_path' (empty path = no reservoir cache, "
        "i.e. genomic chr_train pool). e.g. "
        "'genomic=,motif_planted_v2=outputs/reservoir_cache/...,dinuc_shuffle=...'",
    )
    ap.add_argument(
        "--cross_R_val",
        default="",
        help="Comma list of 'R=val_spec' where val_spec is 'chr_val', a val "
        "cache path, or empty (10pct holdout). Defaults to empty for all.",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Parse R specs
    R_caches = parse_R_spec(args.cross_R)
    R_val = dict(parse_R_spec(args.cross_R_val) if args.cross_R_val else [])
    for R, _ in R_caches:
        R_val.setdefault(R, None)
    R_names = [R for R, _ in R_caches]
    print(f"=== Cross-R pilot over: {R_names} ===")

    # ── Regime stamp (shared across all R) ───────────────────────────────
    esp = args.early_stop_patience
    min_delta = args.min_delta
    battery_prov = load_battery_provenance()
    regime = build_regime(args, esp, min_delta, battery_prov)
    rk = regime_key(regime)

    # ── Load all R data + labels ────────────────────────────────────────
    R_data: dict[str, dict] = {}
    print(f"=== Loading data for {len(R_names)} reservoirs (D={args.D}) ===")
    for R, cache in R_caches:
        print(f"  loading {R} (cache={cache}, val={R_val[R]}) ...", flush=True)
        train_seqs, train_labels, val_seqs, val_labels = load_R_data(
            args.D, args.ref_only, args.data_seed, R, cache, R_val[R]
        )
        R_data[R] = {
            "train_seqs": train_seqs,
            "train_labels": train_labels,
            "val_seqs": val_seqs,
            "val_labels": val_labels,
        }
    test_seqs, test_oracle, test_true = load_chr_test_genomic()
    all_test_sets = load_all_test_sets()

    # Stamp regime + save labels per reservoir (in {out_root}/{R}/labels.npz)
    for R in R_names:
        R_dir = out_root / R
        R_dir.mkdir(parents=True, exist_ok=True)
        regime_path = R_dir / "regime.json"
        if regime_path.exists():
            prior = json.loads(regime_path.read_text())
            if regime_key(prior) != rk:
                raise SystemExit(f"{regime_path} has DIFFERENT regime; use fresh out_dir")
        else:
            _atomic_write_text(regime_path, json.dumps(regime, indent=2))
        # save labels
        label_dict = {
            "val_labels": R_data[R]["val_labels"],
            "test_oracle": test_oracle,
            "test_true": test_true,
        }
        for set_name, (_, ol) in all_test_sets.items():
            label_dict[f"oracle_{set_name}"] = ol
        label_dict["regime_json"] = np.array(json.dumps(regime))
        _atomic_savez(R_dir / "labels.npz", **label_dict)

    # ── Build strategies (single set, fed cross-R mean val) ──────────────
    sys.path.insert(0, str(REPO))
    from experiments.hp_strategies import get_strategy

    try:
        from experiments import llm_autoresearch  # registers llm strategy
    except ImportError:
        pass

    strategy_names = args.strategies.split(",")
    strategies = {
        name: get_strategy(name, seed=args.hp_seed + i * 1000, D=args.D)
        for i, name in enumerate(strategy_names)
    }
    print(f"=== Strategies: {list(strategies)} ===")

    # ── Resume: preload history from any existing R subdirs ──────────────
    n_preloaded = 0
    per_strat_hist: dict[str, tuple[list, list]] = {n: ([], []) for n in strategies}
    for strat_name in strategies:
        # Cell-mean val per (round, hp_repr) across reservoirs
        round_vals: dict[tuple[int, str], list[float]] = {}
        round_hps: dict[tuple[int, str], HPConfig] = {}
        for R in R_names:
            strat_dir = out_root / R / strat_name
            if not strat_dir.exists():
                continue
            for meta_path in sorted(strat_dir.glob("r*_meta.json")):
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    continue
                if regime_key(meta.get("regime")) != rk:
                    continue
                if "val_pearson" not in meta:
                    continue
                hp_fields = {
                    k: v
                    for k, v in meta.get("hp", {}).items()
                    if k in HPConfig.__dataclass_fields__
                }
                hp = HPConfig(**hp_fields)
                key = (int(meta.get("round", -1)), json.dumps(_hp_to_dict(hp), sort_keys=True))
                round_vals.setdefault(key, []).append(float(meta["val_pearson"]))
                round_hps[key] = hp
        for key, vals in round_vals.items():
            if len(vals) == len(R_names):  # only count complete cross-R rounds
                per_strat_hist[strat_name][0].append(round_hps[key])
                per_strat_hist[strat_name][1].append(float(np.mean(vals)))
                n_preloaded += 1
    for name, (cs, vs) in per_strat_hist.items():
        if cs:
            strategies[name].update(cs, vs)
            print(f"  [resume] preloaded {len(cs)} complete cross-R rounds into '{name}'")

    # ── Search loop ──────────────────────────────────────────────────────
    total_models = 0
    t0 = time.time()
    for rd in range(args.rounds):
        print(
            f"\n=== Round {rd + 1}/{args.rounds}  (elapsed {(time.time() - t0) / 60:.1f}m, "
            f"{total_models} models trained) ===",
            flush=True,
        )
        proposals_path = out_root / f"round_{rd:02d}_proposals.json"
        if proposals_path.exists():
            saved = json.loads(proposals_path.read_text())
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
            print(f"  [resume] loaded {len(round_configs)} saved proposals")
        else:
            round_configs = []
            for name, strat in strategies.items():
                configs = strat.suggest(args.per_strategy_per_round)
                for c in configs:
                    round_configs.append((name, c))
                print(f"  {name}: proposed {len(configs)} configs")
            _atomic_write_text(
                proposals_path,
                json.dumps(
                    [{"strategy": n_, "hp": _hp_to_dict(c)} for n_, c in round_configs], indent=2
                ),
            )

        round_cross_R_vals: dict[str, list[float]] = {n: [] for n in strategies}
        round_cross_R_cfgs: dict[str, list[HPConfig]] = {n: [] for n in strategies}

        for i, (strat_name, hp) in enumerate(round_configs):
            model_id = f"r{rd:02d}_{strat_name}_{i:02d}"
            per_R_vals: list[float] = []
            for R in R_names:
                meta_path = out_root / R / strat_name / f"{model_id}_meta.json"
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                if meta_path.exists():
                    try:
                        prior = json.loads(meta_path.read_text())
                    except Exception:
                        prior = {}
                    if "val_pearson" in prior:
                        per_R_vals.append(float(prior["val_pearson"]))
                        continue

                print(
                    f"  Training {model_id} on {R}: lr={hp.lr:.1e} bs={hp.batch_size} "
                    f"layers={hp.n_layers} width={hp.width_base}",
                    flush=True,
                )
                try:
                    result = train_one_model(
                        hp,
                        R_data[R]["train_seqs"],
                        R_data[R]["train_labels"],
                        R_data[R]["val_seqs"],
                        R_data[R]["val_labels"],
                        test_seqs,
                        epochs=args.epochs,
                        use_compile=False,
                        early_stopping_patience=esp,
                        min_delta=min_delta,
                        extra_test_sets=all_test_sets,
                    )
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    result = {
                        "hp": _hp_to_dict(hp),
                        "error": str(e),
                        "strategy": strat_name,
                        "round": rd,
                    }
                result["model_id"] = model_id
                result["strategy"] = strat_name
                result["round"] = rd
                result["regime"] = regime
                result["reservoir"] = R
                _atomic_savez(
                    out_root / R / strat_name / f"{model_id}.npz",
                    **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                )
                meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
                _atomic_write_text(meta_path, json.dumps(meta, indent=2, default=str))
                if "val_pearson" in result and np.isfinite(result["val_pearson"]):
                    per_R_vals.append(float(result["val_pearson"]))
                total_models += 1
                print(
                    f"    val_pearson({R})={result.get('val_pearson', 'ERR')}  "
                    f"time={result.get('train_time_sec', 0):.0f}s",
                    flush=True,
                )

            if len(per_R_vals) == len(R_names):
                cross_mean = float(np.mean(per_R_vals))
                round_cross_R_vals[strat_name].append(cross_mean)
                round_cross_R_cfgs[strat_name].append(hp)
                print(
                    f"  {model_id} CROSS-R MEAN val = {cross_mean:.4f}  "
                    f"(per-R: {['%.3f' % v for v in per_R_vals]})",
                    flush=True,
                )

        # Update each strategy with the cross-R mean signal
        for name, strat in strategies.items():
            cs, vs = round_cross_R_cfgs[name], round_cross_R_vals[name]
            if cs:
                strat.update(cs, vs)

    print(f"\n=== DONE. Trained {total_models} models. Run cross-R analysis on {out_root} ===")


if __name__ == "__main__":
    main()
