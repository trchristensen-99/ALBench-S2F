"""Phase 2 HP-space coverage + quality analysis.

Walks outputs/phase2_unified/D{D}/{strategy}/r*_meta.json and produces:
  - Per-strategy x per-HP histograms (matrix of small multiples)
  - "Stuck" dimensions report (HPs not actually varied by a strategy)
  - Kept-vs-dropped breakdown from cross_strategy_report.json (which models did ElasticNetCV retain?)
  - Per-strategy gpu_hrs and models/hr efficiency

Usage:
    python scripts/analysis/phase2_coverage.py --out_root outputs/phase2_unified --d 30000
    python scripts/analysis/phase2_coverage.py --out_root outputs/phase2_unified --d 30000 --plot

Output: outputs/phase2_unified/D{D}/coverage_report.json + per-dim PNG plots if --plot.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# HP dims to audit. Each spec: (name, dtype, range_or_choices, default_value)
# Numeric → log-binned histogram; categorical → exact counts.
HP_AUDIT_SPECS = {
    "lr": ("log", (1e-5, 1e-2), None),
    "batch_size": ("cat", [32, 64, 128, 256, 512, 1024], 256),
    "conv_dropout": ("uniform", (0.0, 0.3), 0.0),
    "dense_dropout": ("uniform", (0.0, 0.5), 0.0),
    "n_layers": ("int", (2, 12), 6),
    "width_base": ("cat", [16, 32, 64, 128, 256], 64),
    "block_class": ("cat", ["eff", "ag", "plain"], "eff"),
    "ks": ("cat", [3, 5, 7, 9, 11], 5),
    "pct_start": ("cat", [0.1, 0.2, 0.3, 0.4], 0.3),
    "optimizer": ("cat", ["adam", "adamw", "muon"], "adamw"),
    "weight_decay": ("log", (1e-6, 1e-2), None),
    "use_shift_aug": ("cat", [False, True], False),
    "shift_max": ("cat", [5, 10, 15, 20], 15),
    "use_evoaug": ("cat", [False, True], False),
}


def walk_metas(out_root: Path, d_train: int) -> dict[str, list[dict]]:
    """Return {strategy_name: [meta dicts]} for D=d_train."""
    out: dict[str, list[dict]] = defaultdict(list)
    d_dir = out_root / f"D{d_train}"
    if not d_dir.exists():
        raise FileNotFoundError(f"{d_dir} not found")
    for strat_dir in sorted(d_dir.iterdir()):
        if not strat_dir.is_dir():
            continue
        metas = []
        for meta_path in sorted(strat_dir.glob("r*_meta.json")):
            try:
                m = json.loads(meta_path.read_text())
                if "val_pearson" in m or "val_mse" in m:
                    metas.append(m)
            except Exception:
                continue
        if metas:
            out[strat_dir.name] = metas
    return dict(out)


def per_strategy_histogram(strategy_metas: dict[str, list[dict]]) -> dict:
    """For each (strategy, hp), compute distinct values + distribution."""
    report = {}
    for strat, metas in strategy_metas.items():
        sr = {}
        for hp_name, (kind, spec, default) in HP_AUDIT_SPECS.items():
            values = [m.get("hp", {}).get(hp_name) for m in metas]
            values = [v for v in values if v is not None]
            if not values:
                sr[hp_name] = {"kind": kind, "n_present": 0}
                continue
            distinct = sorted({str(v) for v in values})
            entry = {
                "kind": kind,
                "n_present": len(values),
                "n_distinct": len(distinct),
                "distinct_values": distinct,
            }
            if kind == "cat":
                counter = Counter(str(v) for v in values)
                entry["counts"] = dict(counter.most_common())
                non_default = sum(c for v, c in counter.items() if v != str(default))
                entry["non_default_pct"] = round(100 * non_default / len(values), 1)
            else:
                arr = np.array([float(v) for v in values])
                entry["min"] = float(arr.min())
                entry["max"] = float(arr.max())
                entry["median"] = float(np.median(arr))
                entry["range_fraction"] = (
                    float((arr.max() - arr.min()) / (spec[1] - spec[0]))
                    if kind != "log"
                    else float(
                        (np.log10(arr.max()) - np.log10(arr.min()))
                        / (np.log10(spec[1]) - np.log10(spec[0]))
                    )
                )
            sr[hp_name] = entry
        report[strat] = {
            "n_models": len(metas),
            "hps": sr,
            "gpu_hrs": sum(m.get("train_time_sec", 0) for m in metas) / 3600,
        }
    return report


def coverage_gaps(per_strategy_report: dict) -> dict:
    """Identify HPs no strategy varied much (likely stuck at default)."""
    gaps = {}
    for hp_name, (kind, _, default) in HP_AUDIT_SPECS.items():
        # For each strategy, check coverage of this HP
        strat_distinct = {
            s: r["hps"].get(hp_name, {}).get("n_distinct", 0)
            for s, r in per_strategy_report.items()
        }
        max_distinct = max(strat_distinct.values()) if strat_distinct else 0
        # Stuck = ≤1 distinct in all strategies, OR <30% non-default pct on cat HPs
        stuck_strategies = []
        for s, r in per_strategy_report.items():
            hp_info = r["hps"].get(hp_name, {})
            if hp_info.get("n_distinct", 0) <= 1:
                stuck_strategies.append(s)
            elif kind == "cat" and hp_info.get("non_default_pct", 100) < 30:
                stuck_strategies.append(s)
        if stuck_strategies:
            gaps[hp_name] = {
                "max_distinct_any_strategy": max_distinct,
                "stuck_strategies": stuck_strategies,
            }
    return gaps


def cross_strategy_report(out_root: Path, d_train: int) -> dict:
    csr_path = out_root / f"D{d_train}" / "cross_strategy_report.json"
    if csr_path.exists():
        return json.loads(csr_path.read_text())
    return {}


def kept_models_breakdown(per_strategy_report: dict, cross_report: dict) -> dict:
    """Per-strategy: how many models did ElasticNetCV retain in the unified ensemble."""
    if not cross_report:
        return {}
    out = {}
    for strat, r in per_strategy_report.items():
        info = cross_report.get("per_strategy", {}).get(strat, {})
        kept = info.get("enet_kept", info.get("n_kept", None))
        total = r["n_models"]
        out[strat] = {
            "trained": total,
            "kept_in_self_ensemble": kept,
            "kept_pct": round(100 * kept / total, 1) if kept is not None and total else None,
            "gpu_hrs": round(r["gpu_hrs"], 1),
            "gpu_hrs_per_kept": round(r["gpu_hrs"] / kept, 2) if kept else None,
        }
    return out


def maybe_plot(per_strategy_report: dict, out_dir: Path) -> None:
    """Render small-multiples histogram per (strategy, hp). Saves to out_dir/coverage_hist.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strats = sorted(per_strategy_report.keys())
    hps = list(HP_AUDIT_SPECS.keys())
    fig, axes = plt.subplots(
        len(hps), len(strats), figsize=(2.2 * len(strats), 1.6 * len(hps)), squeeze=False
    )
    for i, hp in enumerate(hps):
        kind, spec, _ = HP_AUDIT_SPECS[hp]
        for j, strat in enumerate(strats):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            entry = per_strategy_report[strat]["hps"].get(hp, {})
            if not entry or entry.get("n_present", 0) == 0:
                ax.text(
                    0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes, color="gray"
                )
                continue
            if kind == "cat":
                counts = entry.get("counts", {})
                labels = [str(v) for v in spec]
                values = [counts.get(lbl, 0) for lbl in labels]
                ax.bar(
                    range(len(labels)), values, color="steelblue", edgecolor="black", linewidth=0.5
                )
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
            elif kind == "log":
                # Reconstruct values list from distinct (approximate; if needed, re-walk)
                # For simplicity use {distinct_values, n_present} — approximate viz
                ax.text(
                    0.5,
                    0.5,
                    f"min={entry['min']:.1e}\nmax={entry['max']:.1e}\nspan={entry['range_fraction']:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    transform=ax.transAxes,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"min={entry['min']:.2g}\nmax={entry['max']:.2g}\nspan={entry.get('range_fraction', 0):.0%}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    transform=ax.transAxes,
                )
            if i == 0:
                ax.set_title(strat, fontsize=7)
            if j == 0:
                ax.set_ylabel(hp, fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Phase 2 HP coverage per strategy", fontsize=10)
    plt.tight_layout()
    out_path = out_dir / "coverage_hist.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  plot saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="e.g. outputs/phase2_unified")
    ap.add_argument("--d", type=int, default=30000)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    print(f"Walking {out_root}/D{args.d} …")
    metas = walk_metas(out_root, args.d)
    print(f"  found {len(metas)} strategies, {sum(len(v) for v in metas.values())} models")

    per_strat = per_strategy_histogram(metas)
    gaps = coverage_gaps(per_strat)
    csr = cross_strategy_report(out_root, args.d)
    kept = kept_models_breakdown(per_strat, csr)

    report = {
        "d_train": args.d,
        "n_strategies": len(metas),
        "n_models_total": sum(r["n_models"] for r in per_strat.values()),
        "per_strategy": per_strat,
        "coverage_gaps": gaps,
        "kept_breakdown": kept,
    }

    out_path = out_root / f"D{args.d}" / "coverage_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  report → {out_path}")

    # Print concise summary
    print()
    print("=== Coverage gaps (HPs not exercised by some strategies) ===")
    for hp, info in gaps.items():
        print(
            f"  {hp:<16}: max_distinct_any={info['max_distinct_any_strategy']:>2}  "
            f"stuck_in={len(info['stuck_strategies'])} strategies"
        )
    print()
    print("=== Per-strategy GPU efficiency ===")
    print(f"  {'strategy':<24} {'n':>4} {'kept':>4} {'kept%':>6} {'gpu_hr':>8} {'gpu_hr/kept':>12}")
    for s, info in sorted(kept.items(), key=lambda x: -(x[1].get("kept_in_self_ensemble") or 0)):
        kept_n = info.get("kept_in_self_ensemble", "—")
        kept_pct = info.get("kept_pct", "—")
        gh = info.get("gpu_hrs", "—")
        gpk = info.get("gpu_hrs_per_kept", "—")
        print(f"  {s:<24} {info['trained']:>4} {kept_n!s:>4} {kept_pct!s:>6} {gh!s:>8} {gpk!s:>12}")

    if args.plot:
        maybe_plot(per_strat, out_root / f"D{args.d}")


if __name__ == "__main__":
    main()
