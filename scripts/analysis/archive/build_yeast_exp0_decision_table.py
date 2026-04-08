#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _mean(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _extract_test(metrics: dict, key: str) -> float:
    return float(metrics.get(key, {}).get("pearson_r", np.nan))


def _dream_scaling_rows(root: Path, label: str) -> list[dict]:
    rows = []
    by_frac: dict[float, list[dict]] = {}
    for p in root.glob("seed_*/fraction_*/result.json"):
        d = json.loads(p.read_text())
        f = float(d.get("fraction"))
        by_frac.setdefault(f, []).append(d)

    for frac in sorted(by_frac):
        vals = by_frac[frac]
        id_vals = [_extract_test(v.get("test_metrics", {}), "random") for v in vals]
        ood_vals = [_extract_test(v.get("test_metrics", {}), "genomic") for v in vals]
        snv_vals = [_extract_test(v.get("test_metrics", {}), "snv_abs") for v in vals]
        val_vals = [float(v.get("best_val_pearson_r", np.nan)) for v in vals]
        row = {
            "group": label,
            "name": f"scaling_frac_{frac:.3f}",
            "n_runs": len(vals),
            "val_pearson": _mean(val_vals),
            "id_pearson": _mean(id_vals),
            "ood_pearson": _mean(ood_vals),
            "snv_abs_pearson": _mean(snv_vals),
        }
        row["id_minus_ood"] = row["id_pearson"] - row["ood_pearson"]
        rows.append(row)
    return rows


def _dream_oracle_row(root: Path) -> dict | None:
    vals = []
    for p in sorted(root.glob("oracle_*/summary.json")):
        d = json.loads(p.read_text())
        tm = d.get("test_metrics", {})
        vals.append(
            {
                "val": float(d.get("best_val_pearson_r", np.nan)),
                "id": _extract_test(tm, "random"),
                "ood": _extract_test(tm, "genomic"),
                "snv": _extract_test(tm, "snv_abs"),
            }
        )
    if not vals:
        return None
    row = {
        "group": "dream_oracle",
        "name": "kfold_ensemble_mean",
        "n_runs": len(vals),
        "val_pearson": _mean([v["val"] for v in vals]),
        "id_pearson": _mean([v["id"] for v in vals]),
        "ood_pearson": _mean([v["ood"] for v in vals]),
        "snv_abs_pearson": _mean([v["snv"] for v in vals]),
    }
    row["id_minus_ood"] = row["id_pearson"] - row["ood_pearson"]
    return row


def _ag_finetune_rows(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.glob("*/summary.json")):
        d = json.loads(p.read_text())
        tm = d.get("test_metrics", {})
        row = {
            "group": "ag_finetune",
            "name": p.parent.name,
            "n_runs": 1,
            "val_pearson": float(d.get("best_val_pearson_r", np.nan)),
            "id_pearson": _extract_test(tm, "random"),
            "ood_pearson": _extract_test(tm, "genomic"),
            "snv_abs_pearson": _extract_test(tm, "snv_abs"),
            "second_stage_enabled": bool(d.get("second_stage_enabled", False)),
            "second_stage_unfreeze_mode": d.get("second_stage_unfreeze_mode"),
            "second_stage_max_shift": d.get("second_stage_max_shift"),
        }
        row["id_minus_ood"] = row["id_pearson"] - row["ood_pearson"]
        rows.append(row)
    return rows


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "outputs" / "analysis" / "yeast_exp0_decision_table"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    rows.extend(_dream_scaling_rows(repo / "outputs" / "exp0_yeast_scaling", "dream_scaling_v160"))
    rows.extend(
        _dream_scaling_rows(repo / "outputs" / "exp0_yeast_scaling_v256", "dream_scaling_v256")
    )
    oracle = _dream_oracle_row(repo / "outputs" / "oracle_dream_rnn_yeast_kfold")
    if oracle is not None:
        rows.append(oracle)
    oracle_new = _dream_oracle_row(repo / "outputs" / "oracle_dream_rnn_yeast_kfold_v256")
    if oracle_new is not None:
        oracle_new["group"] = "dream_oracle_v256"
        rows.append(oracle_new)
    rows.extend(_ag_finetune_rows(repo / "outputs" / "ag_yeast_oracle_finetune"))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows found.")
        return

    metric_cols = ["val_pearson", "id_pearson", "ood_pearson", "snv_abs_pearson", "id_minus_ood"]
    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Decision score: prioritize OOD then SNV then ID then val; penalize large ID-OOD gap.
    df["decision_score"] = (
        3.0 * df["ood_pearson"].fillna(0.0)
        + 2.0 * df["snv_abs_pearson"].fillna(0.0)
        + 1.5 * df["id_pearson"].fillna(0.0)
        + 1.0 * df["val_pearson"].fillna(0.0)
        - 0.5 * df["id_minus_ood"].fillna(0.0).clip(lower=0.0)
    )

    df_sorted = df.sort_values(["group", "decision_score"], ascending=[True, False])
    df_sorted.to_csv(out_dir / "decision_table.csv", index=False)

    # Compact top-table for AG finetune selection.
    ag = df_sorted[df_sorted["group"] == "ag_finetune"].copy()
    if not ag.empty:
        ag = ag.sort_values("decision_score", ascending=False)
        ag.to_csv(out_dir / "ag_finetune_ranking.csv", index=False)

    md = ["# Yeast Exp0 Decision Table", "", "Generated from local outputs.", ""]
    show_cols = [
        "group",
        "name",
        "n_runs",
        "val_pearson",
        "id_pearson",
        "ood_pearson",
        "snv_abs_pearson",
        "id_minus_ood",
        "decision_score",
    ]
    md.append(df_sorted[show_cols].round(4).to_markdown(index=False))
    (out_dir / "decision_table.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {out_dir / 'decision_table.csv'}")
    print(f"Wrote: {out_dir / 'decision_table.md'}")
    if (out_dir / "ag_finetune_ranking.csv").exists():
        print(f"Wrote: {out_dir / 'ag_finetune_ranking.csv'}")


if __name__ == "__main__":
    main()
