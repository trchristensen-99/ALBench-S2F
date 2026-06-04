"""Bootstrap CI for ensemble metrics by resampling models within each cell.

For each cell:
  - Loads N (~30) model val_preds + per-set test_preds + labels
  - Resamples N models with replacement B=200 times
  - For each resample: refits ElasticNetCV on (V_resampled, val_labels), applies
    ensemble weights to each test panel's predictions, computes Pearson + MSE
  - Outputs {cell_key: {panel: {pearson_mean, pearson_std, pearson_ci_lo, pearson_ci_hi,
                                mse_mean, mse_std, mse_ci_lo, mse_ci_hi}}}

Run:
  python scripts/bootstrap_ensemble_uncertainty.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet, ElasticNetCV

B = 100  # bootstrap iterations
RESERVOIRS = [
    "genomic",
    "random",
    "prm_1pct",
    "prm_10pct",
    "prm_uncertainty_1pct",
    "evoaug_heavy",
    "motif_shuffled",
    "motif_planted_v2",
]
DS = [300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
RNG_SEED = 42


def find_cell(reservoir: str, D: int) -> Path | None:
    if reservoir == "genomic":
        p = Path(f"outputs/full_sweep_chrval/k562_genomic_d{D}_seed42")
        if (p / "summary.json").exists():
            return p
    p = Path(f"outputs/full_sweep/k562_{reservoir}_d{D}_seed42")
    return p if (p / "summary.json").exists() else None


def load_models(cell: Path) -> tuple[np.ndarray, dict, dict]:
    """Return (V, labels_dict, test_preds_by_set).

    V: (n_val, n_models)
    test_preds_by_set: {set_name: (n_test, n_models) array}
    """
    labels = None
    val_preds = []
    test_preds_by_set: dict[str, list] = {}
    # canonical sets we want
    wanted = [
        "genomic",
        "ood",
        "snv_ref",
        "snv_alt",
        "sub_high",
        "random_32k",
        "dinuc_shuffle",
        "translocation",
        "inversion",
    ]

    for sub in sorted(cell.iterdir()):
        if not sub.is_dir():
            continue
        lab_path = sub / "labels.npz"
        if labels is None and lab_path.exists():
            labels = dict(np.load(lab_path, allow_pickle=True))
        for npz in sorted(sub.glob("r*.npz")):
            if "meta" in npz.name:
                continue
            z = np.load(npz, allow_pickle=True)
            if "val_pred" not in z.files:
                continue
            val_preds.append(z["val_pred"].astype(np.float32))
            for s in wanted:
                key = f"test_pred_{s}"
                if key in z.files:
                    test_preds_by_set.setdefault(s, []).append(z[key].astype(np.float32))
                elif s == "genomic" and "test_pred" in z.files and key not in z.files:
                    test_preds_by_set.setdefault(s, []).append(z["test_pred"].astype(np.float32))

    if labels is None or not val_preds:
        return None, None, None
    V = np.stack(val_preds, axis=1)
    # Drop models with NaN val_pred (broken training)
    finite_models = np.all(np.isfinite(V), axis=0)
    keep_idx = np.where(finite_models)[0]
    if len(keep_idx) < len(val_preds):
        V = V[:, keep_idx]
        test_preds_by_set = {s: [lst[i] for i in keep_idx] for s, lst in test_preds_by_set.items()}
    # Stack per-set preds, dropping sets where coverage is incomplete or any model is NaN
    test_preds = {}
    for s, lst in test_preds_by_set.items():
        if len(lst) != V.shape[1]:
            continue
        T = np.stack(lst, axis=1)
        # Replace NaN/inf in test preds with column mean (safer than dropping models)
        if not np.all(np.isfinite(T)):
            for j in range(T.shape[1]):
                col = T[:, j]
                bad = ~np.isfinite(col)
                if bad.any():
                    col[bad] = np.nanmean(col[~bad]) if (~bad).any() else 0.0
                    T[:, j] = col
        test_preds[s] = T
    return V, labels, test_preds


def bootstrap_cell(V: np.ndarray, labels: dict, test_preds: dict) -> dict:
    """Bootstrap-resample models and compute metrics per panel + snv_delta.

    SPEED OPTIMIZATION: fit ElasticNetCV ONCE on the full model set to find a
    good (alpha, l1_ratio); reuse those for fixed-alpha ElasticNet inside the
    bootstrap loop (~50× speedup vs full CV per bootstrap).
    """
    val_y = labels["val_labels"].astype(np.float32)
    n_models = V.shape[1]
    rng = np.random.default_rng(RNG_SEED)

    # One-shot CV fit on full model set to determine alpha / l1_ratio
    enet_cv = ElasticNetCV(positive=True, cv=3, max_iter=2000, random_state=42, n_alphas=10)
    enet_cv.fit(V, val_y)
    best_alpha = float(enet_cv.alpha_)
    best_l1 = (
        float(enet_cv.l1_ratio_) if np.ndim(enet_cv.l1_ratio_) == 0 else float(enet_cv.l1_ratio_[0])
    )

    # Pre-sample bootstrap indices
    indices = [rng.integers(0, n_models, size=n_models) for _ in range(B)]

    # Refit ElasticNet (fixed alpha) once per resample, reuse coef across all panels
    refitted = []  # list of (coef, intercept)
    for idx in indices:
        try:
            en = ElasticNet(
                alpha=best_alpha, l1_ratio=best_l1, positive=True, max_iter=2000, random_state=42
            )
            en.fit(V[:, idx], val_y)
            refitted.append((idx, en.coef_, float(en.intercept_)))
        except Exception:
            continue

    out = {}
    # Per-panel evaluation
    for panel, T in test_preds.items():
        oracle_key = f"oracle_{panel}"
        if oracle_key not in labels:
            if panel == "genomic" and "test_oracle" in labels:
                y = labels["test_oracle"].astype(np.float32)
            else:
                continue
        else:
            y = labels[oracle_key].astype(np.float32)
        pearsons, mses = [], []
        for idx, coef, intercept in refitted:
            pred = T[:, idx] @ coef + intercept
            mask = np.isfinite(pred) & np.isfinite(y)
            if mask.sum() < 8:
                continue
            r = pearsonr(pred[mask], y[mask])[0]
            mse = ((pred[mask] - y[mask]) ** 2).mean()
            if np.isfinite(r) and np.isfinite(mse):
                pearsons.append(float(r))
                mses.append(float(mse))
        if pearsons:
            out[panel] = _summarize(pearsons, mses)

    # SNV delta panel
    if "snv_ref" in test_preds and "snv_alt" in test_preds:
        ref_oracle = labels.get("oracle_snv_ref")
        alt_oracle = labels.get("oracle_snv_alt")
        if ref_oracle is not None and alt_oracle is not None:
            y_delta = alt_oracle.astype(np.float32) - ref_oracle.astype(np.float32)
            ref_T = test_preds["snv_ref"]
            alt_T = test_preds["snv_alt"]
            pearsons, mses = [], []
            for idx, coef, intercept in refitted:
                pred_ref = ref_T[:, idx] @ coef + intercept
                pred_alt = alt_T[:, idx] @ coef + intercept
                pred_d = pred_alt - pred_ref
                mask = np.isfinite(pred_d) & np.isfinite(y_delta)
                if mask.sum() < 8:
                    continue
                r = pearsonr(pred_d[mask], y_delta[mask])[0]
                mse = ((pred_d[mask] - y_delta[mask]) ** 2).mean()
                if np.isfinite(r) and np.isfinite(mse):
                    pearsons.append(float(r))
                    mses.append(float(mse))
            if pearsons:
                out["snv_delta"] = _summarize(pearsons, mses)

    return out


def _summarize(pearsons, mses):
    p = np.array(pearsons)
    m = np.array(mses)
    return {
        "pearson_mean": float(p.mean()),
        "pearson_std": float(p.std(ddof=1)),
        "pearson_ci_lo": float(np.percentile(p, 2.5)),
        "pearson_ci_hi": float(np.percentile(p, 97.5)),
        "mse_mean": float(m.mean()),
        "mse_std": float(m.std(ddof=1)),
        "mse_ci_lo": float(np.percentile(m, 2.5)),
        "mse_ci_hi": float(np.percentile(m, 97.5)),
        "n_bootstraps": int(len(p)),
    }


def main():
    results = {}
    n_cells = len(RESERVOIRS) * len(DS)
    done = 0
    for r in RESERVOIRS:
        for D in DS:
            done += 1
            cell = find_cell(r, D)
            if cell is None:
                print(f"  [{done}/{n_cells}] no cell: {r} D={D}")
                continue
            V, labels, tp = load_models(cell)
            if V is None or V.shape[1] < 6:
                print(f"  [{done}/{n_cells}] insufficient models: {r} D={D}")
                continue
            print(f"  [{done}/{n_cells}] {r} D={D}  n_models={V.shape[1]}", flush=True)
            res = bootstrap_cell(V, labels, tp)
            results[f"{r}|{D}"] = {
                "reservoir": r,
                "D": D,
                "n_models": int(V.shape[1]),
                "panels": res,
            }

    out = Path("outputs/poster_plots/bootstrap_uncertainty.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} cells)")


if __name__ == "__main__":
    main()
