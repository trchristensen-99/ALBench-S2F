"""Fit ElasticNetCV ensemble across config_{A,C,D,E}/model.npz for each
(reservoir, D, seed) cell in outputs/focused_train/. Save ensemble predictions
+ summary.json for plotting.

Runs idempotently — skips cells without all configs present OR with summary.json
already written. Safe to call repeatedly in a watchdog loop.

Output per cell:
  outputs/focused_train/k562_{r}_d{D}_seed{S}/summary.json
  outputs/focused_train/k562_{r}_d{D}_seed{S}/ensemble.npz  (val + per-panel test preds)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV

CONFIGS = ["A", "C", "D", "E"]
ROOT = Path("outputs/focused_train")


def panel_oracle_key(panel: str) -> str:
    return f"oracle_{panel}"


def analyze_cell(cell_dir: Path) -> bool:
    summary_path = cell_dir / "summary.json"
    if summary_path.exists():
        return False  # already done

    # Discover available configs
    model_files = []
    config_ids = []
    for cfg in CONFIGS:
        f = cell_dir / f"config_{cfg}" / "model.npz"
        if f.exists():
            model_files.append(f)
            config_ids.append(cfg)
    if len(model_files) < 2:
        return False  # not enough configs to ensemble

    # Load labels (any config's labels.npz)
    labels_path = cell_dir / f"config_{config_ids[0]}" / "labels.npz"
    if not labels_path.exists():
        return False
    labels = dict(np.load(labels_path, allow_pickle=True))
    val_labels = labels["val_labels"].astype(np.float32)

    # Stack val + per-panel test predictions
    val_preds = []
    test_preds_per_panel: dict[str, list[np.ndarray]] = {}
    panel_keys = [k.removeprefix("oracle_") for k in labels if k.startswith("oracle_")]
    panel_keys.append("genomic")  # ensure genomic present (legacy)
    panel_keys = list(set(panel_keys))

    for f in model_files:
        z = np.load(f, allow_pickle=True)
        val_preds.append(z["val_pred"].astype(np.float32))
        for panel in panel_keys:
            key = f"test_pred_{panel}"
            if key in z.files:
                test_preds_per_panel.setdefault(panel, []).append(z[key].astype(np.float32))

    V = np.stack(val_preds, axis=1)  # (n_val, n_models)
    # Drop columns with NaN
    finite_cols = np.all(np.isfinite(V), axis=0)
    if finite_cols.sum() < 2:
        return False
    V = V[:, finite_cols]
    kept_configs = [c for c, keep in zip(config_ids, finite_cols) if keep]
    test_preds_per_panel = {
        p: [lst[i] for i, k in enumerate(finite_cols) if k]
        for p, lst in test_preds_per_panel.items()
        if len(lst) == len(config_ids)
    }

    # Fit ElasticNetCV (positive=True)
    enet = ElasticNetCV(positive=True, cv=3, max_iter=5000, random_state=42, n_alphas=10)
    enet.fit(V, val_labels)
    coef = enet.coef_
    intercept = float(enet.intercept_)
    n_kept = int((coef > 0).sum())

    # Val metrics
    val_ens = V @ coef + intercept
    val_pearson = float(pearsonr(val_ens, val_labels)[0])
    val_mse = float(((val_ens - val_labels) ** 2).mean())

    # Per-panel evaluation
    per_set = {}
    ensemble_data = {"val_pred": val_ens.astype(np.float32), "val_labels": val_labels}
    for panel, preds_list in test_preds_per_panel.items():
        if len(preds_list) != V.shape[1]:
            continue
        T = np.stack(preds_list, axis=1)
        ens = T @ coef + intercept
        oracle_key = panel_oracle_key(panel)
        if oracle_key in labels:
            y = labels[oracle_key].astype(np.float32)
        elif panel == "genomic" and "test_oracle" in labels:
            y = labels["test_oracle"].astype(np.float32)
        else:
            continue
        mask = np.isfinite(ens) & np.isfinite(y)
        if mask.sum() < 8:
            continue
        r = float(pearsonr(ens[mask], y[mask])[0])
        mse = float(((ens[mask] - y[mask]) ** 2).mean())
        per_set[panel] = {"pearson": r, "mse": mse, "n": int(mask.sum())}
        ensemble_data[f"ensemble_test_pred_{panel}"] = ens.astype(np.float32)

    # SNV delta panel (alt − ref)
    snv_delta_metrics = {}
    if "snv_ref" in test_preds_per_panel and "snv_alt" in test_preds_per_panel:
        oracle_ref = labels.get("oracle_snv_ref")
        oracle_alt = labels.get("oracle_snv_alt")
        if oracle_ref is not None and oracle_alt is not None:
            T_ref = np.stack(test_preds_per_panel["snv_ref"], axis=1)
            T_alt = np.stack(test_preds_per_panel["snv_alt"], axis=1)
            ens_ref = T_ref @ coef + intercept
            ens_alt = T_alt @ coef + intercept
            ens_d = ens_alt - ens_ref
            y_d = oracle_alt.astype(np.float32) - oracle_ref.astype(np.float32)
            mask = np.isfinite(ens_d) & np.isfinite(y_d)
            if mask.sum() >= 8:
                r = float(pearsonr(ens_d[mask], y_d[mask])[0])
                mse = float(((ens_d[mask] - y_d[mask]) ** 2).mean())
                snv_delta_metrics["oracle"] = {"pearson": r, "mse": mse, "n": int(mask.sum())}
                ensemble_data["ensemble_snv_delta_pred"] = ens_d.astype(np.float32)
                ensemble_data["snv_delta_label"] = y_d.astype(np.float32)

    summary = {
        "cell_dir": str(cell_dir),
        "configs_kept": kept_configs,
        "n_models_total": int(V.shape[1]),
        "n_models_kept": n_kept,
        "ensemble_coef_sum": float(coef.sum()),
        "val_pearson": val_pearson,
        "val_mse": val_mse,
        "per_set": per_set,
        "snv_delta": snv_delta_metrics,
        "ensemble_coefs": {c: float(w) for c, w in zip(kept_configs, coef.tolist())},
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    np.savez(cell_dir / "ensemble.npz", **ensemble_data)
    return True


def main():
    if not ROOT.exists():
        print(f"No focused_train output dir yet at {ROOT}")
        return
    n_new = 0
    n_skipped = 0
    n_partial = 0
    for cell_dir in sorted(ROOT.iterdir()):
        if not cell_dir.is_dir():
            continue
        result = analyze_cell(cell_dir)
        if result is True:
            n_new += 1
        elif (cell_dir / "summary.json").exists():
            n_skipped += 1
        else:
            n_partial += 1
    print(f"  analyzed {n_new}  already_done {n_skipped}  partial/insufficient {n_partial}")


if __name__ == "__main__":
    main()
