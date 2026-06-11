"""PHASE-4 DEPLOY trainer — NO SEARCH.

Train a FROZEN list of N* HP configs (the recipe locked by Phase 3) on ONE cell
(reservoir x acquisition x D x seed), then ensemble the trained members with
ElasticNetCV REFIT on THIS cell's own val predictions.

WHY a separate launcher: scaling_hp_search.py runs a search loop (strategies propose
configs round by round). At deploy there is nothing left to search — the architectures
are fixed and shared across every reservoir x acquisition x D combo. We only train them
and blend them. This reuses scaling_hp_search's data loaders + train_one_model + the
exact per-model output contract (*_meta.json + sibling .npz + labels.npz), so every
existing analysis script reads a deploy cell unchanged.

THE INVARIANT (why we don't carry frozen weights): the N* member ARCHITECTURES are
shared, but the ENSEMBLE WEIGHTS are refit per cell via
albench.ensemble.fit_elasticnet_stack — never frozen — because the same configs are
worth different amounts on different reservoir/acquisition landscapes. The blend is fit
on the cell's own held-out val preds and applied (same coefficients) to every test set.

RECIPE FORMAT (--recipe): JSON holding a list of FULL HP-config dicts (the keys of
HPConfig: lr, batch_size, n_layers, width_base, block_class, optimizer, ...). Accepts a
bare list, or {"configs": [...]} / {"recipe": [...]}. Each dict is filtered to HPConfig
fields, so extra provenance keys (id, strategy, sig) are ignored. The aggregation step
that emits the final recipe must include the full hp dicts of the chosen models (copy
each chosen model's meta['hp']), NOT just ids.

Resume-safe: a member whose *_meta.json carries a real val_pearson is skipped. The
ensemble step runs only once every member is complete; .deploy_done is the sentinel.

Usage:
  python experiments/deploy_train.py --recipe recipe.json --out_dir <cell> \\
    --D 30000 --data_seed 42 --ref_only --reservoir_cache <pool.npz> \\
    --reservoir_val_cache <val.npz> --epochs 100
  # genomic cell: drop --reservoir_cache/--reservoir_val_cache, add --chr_val
"""

from __future__ import annotations

import argparse
import json
from dataclasses import MISSING
from pathlib import Path

import numpy as np

from albench.ensemble import fit_elasticnet_stack
from experiments.scaling_hp_search import (
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


def _load_recipe(path: Path) -> list[HPConfig]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        items = raw.get("configs") or raw.get("recipe") or raw.get("chosen") or []
    else:
        items = raw
    if not items:
        raise SystemExit(f"recipe {path} holds no configs")
    configs = []
    for i, d in enumerate(items):
        hp_dict = d.get("hp", d) if isinstance(d, dict) else d
        kw = {k: v for k, v in hp_dict.items() if k in HPConfig.__dataclass_fields__}
        missing = [
            f
            for f, fld in HPConfig.__dataclass_fields__.items()
            if f not in kw and fld.default is MISSING and fld.default_factory is MISSING
        ]
        if missing:
            raise SystemExit(f"recipe config {i} missing required HP fields: {missing}")
        configs.append(HPConfig(**kw))
    return configs


def _ensemble_cell(out_dir: Path, members: list[str], labels: dict, all_test_sets: dict) -> None:
    """Refit the per-cell ElasticNet blend on val preds; apply to every test set."""
    val_y = labels["val_labels"].astype(np.float64)
    val_X, test_X, ids = [], [], []
    for mid in members:
        d = np.load(out_dir / f"{mid}.npz")
        if "val_pred" not in d.files or "test_pred" not in d.files:
            continue
        val_X.append(d["val_pred"].astype(np.float64))
        test_X.append(d["test_pred"].astype(np.float64))
        ids.append(mid)
    if len(val_X) < 1:
        raise SystemExit("no trained members with predictions to ensemble")
    val_X = np.vstack(val_X)
    test_X = np.vstack(test_X)

    test_pred, _vpred, info, enet = fit_elasticnet_stack(
        val_X, val_y, test_X, return_info=True, return_estimator=True
    )

    from scipy.stats import pearsonr

    def _metrics(pred, target):
        m = np.isfinite(pred) & np.isfinite(target)
        if m.sum() < 8:
            return {"pearson": None, "mse": None, "n": int(m.sum())}
        return {
            "pearson": float(pearsonr(pred[m], target[m])[0]),
            "mse": float(((pred[m] - target[m]) ** 2).mean()),
            "n": int(m.sum()),
        }

    per_set = {
        "genomic_test_oracle": _metrics(test_pred, labels["test_oracle"].astype(np.float64)),
        "genomic_test_true": _metrics(test_pred, labels["test_true"].astype(np.float64)),
    }
    # Apply the SAME blend to each battery test set (members store test_pred_<set>).
    for set_name in all_test_sets:
        key = f"test_pred_{set_name}"
        mats = []
        ok = True
        for mid in ids:
            d = np.load(out_dir / f"{mid}.npz")
            if key not in d.files:
                ok = False
                break
            mats.append(d[key].astype(np.float64))
        oracle_key = f"oracle_{set_name}"
        if not ok or oracle_key not in labels:
            continue
        set_pred = enet.predict(np.vstack(mats).T)
        per_set[set_name] = _metrics(set_pred, labels[oracle_key].astype(np.float64))

    blend = {
        "members": ids,
        "weights": {mid: float(w) for mid, w in zip(ids, info["coef"])},
        "intercept": info["intercept"],
        "n_kept": info["n_kept"],
        "alpha": info["alpha"],
        "l1_ratio": info["l1_ratio"],
        "note": "ElasticNetCV(positive=True) refit on THIS cell's own val preds — not frozen.",
    }
    out = {"blend": blend, "metrics": per_set}
    _atomic_write_text(out_dir / "deploy_ensemble.json", json.dumps(out, indent=2, default=str))
    print(f"=== deploy ensemble: {info['n_kept']}/{len(ids)} members kept ===")
    for s, m in per_set.items():
        if m["pearson"] is not None:
            print(f"  {s:24s} pearson={m['pearson']:.4f} mse={m['mse']:.4f} (n={m['n']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, help="JSON with the frozen N* HP configs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--D", type=int, default=30000)
    ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--ref_only", action="store_true")
    ap.add_argument("--reservoir_cache", default=None)
    ap.add_argument("--chr_val", action="store_true")
    ap.add_argument("--reservoir_val_cache", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--min_delta", type=float, default=1e-3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = _load_recipe(Path(args.recipe))
    print(f"=== Deploy: {len(configs)} frozen configs -> {out_dir} ===")

    esp = args.early_stop_patience or 15
    regime = build_regime(args, esp, args.min_delta, load_battery_provenance())
    rk = regime_key(regime)
    regime_path = out_dir / "regime.json"
    if regime_path.exists():
        prior = json.loads(regime_path.read_text())
        if regime_key(prior) != rk:
            raise SystemExit(f"{out_dir} holds a DIFFERENT regime; use a fresh --out_dir.")
    else:
        _atomic_write_text(regime_path, json.dumps(regime, indent=2))

    train_seqs, train_labels, val_seqs, val_labels = load_chr_train_pool(
        args.D,
        ref_only=args.ref_only,
        val_frac=0.1,
        seed=args.data_seed,
        reservoir_cache=args.reservoir_cache,
        chr_val=args.chr_val,
        reservoir_val_cache=args.reservoir_val_cache,
    )
    test_seqs, test_oracle, test_true = load_chr_test_genomic()
    all_test_sets = load_all_test_sets()

    label_dict = {
        "val_labels": val_labels,
        "test_oracle": test_oracle,
        "test_true": test_true,
    }
    for set_name, (_, oracle_labels) in all_test_sets.items():
        label_dict[f"oracle_{set_name}"] = oracle_labels
    label_dict["regime_json"] = np.array(json.dumps(regime))
    _atomic_savez(out_dir / "labels.npz", **label_dict)
    labels = {k: label_dict[k] for k in ("val_labels", "test_oracle", "test_true")}
    for set_name in all_test_sets:
        labels[f"oracle_{set_name}"] = label_dict[f"oracle_{set_name}"]

    members = []
    for i, hp in enumerate(configs):
        model_id = f"deploy_{i:02d}"
        members.append(model_id)
        meta_path = out_dir / f"{model_id}_meta.json"
        if meta_path.exists():
            try:
                prior_meta = json.loads(meta_path.read_text())
            except Exception:
                prior_meta = {}
            if regime_key(prior_meta.get("regime")) != rk:
                raise SystemExit(f"{meta_path} is from a different regime; use a fresh --out_dir.")
            if "val_pearson" in prior_meta:
                print(f"  [resume] skip {model_id} (already done)")
                continue
        print(f"\n  Training {model_id}: lr={hp.lr:.1e} layers={hp.n_layers} opt={hp.optimizer}")
        try:
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
                min_delta=args.min_delta,
                extra_test_sets=all_test_sets,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            result = {"hp": _hp_to_dict(hp), "error": str(e)}
        result["model_id"] = model_id
        result["strategy"] = "deploy"
        result["regime"] = regime
        _atomic_savez(
            out_dir / f"{model_id}.npz",
            **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
        )
        meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        _atomic_write_text(
            out_dir / f"{model_id}_meta.json", json.dumps(meta, indent=2, default=str)
        )
        print(f"    val_pearson={result.get('val_pearson', 'ERR')}")

    done = [
        m
        for m in members
        if (out_dir / f"{m}_meta.json").exists()
        and "val_pearson" in json.loads((out_dir / f"{m}_meta.json").read_text())
    ]
    if len(done) < len(members):
        raise SystemExit(
            f"only {len(done)}/{len(members)} members complete; re-run to finish before ensembling."
        )
    _ensemble_cell(out_dir, members, labels, all_test_sets)
    (out_dir / ".deploy_done").touch()
    print(f"\n=== DEPLOY DONE: {len(members)} members + per-cell ElasticNet blend ===")


if __name__ == "__main__":
    main()
