"""Multi-D extension of analyze_hp_flatness.

Reads task3_lr_bs/ (D=600k) AND task3b_lr_bs_d{500,6000,60000}/ to plot
per-D LR×BS heatmaps for each arch. Highlights the optimum at each D
plus the locked optimum from D=600k. If the per-D optimum drifts as D
changes, that's scale coupling — flagged.

Outputs:
    results/preflight/figures/task3b_heatmaps_<arch>.png
        2×2 panel of LR×BS heatmaps at D ∈ {500, 6k, 60k, 600k}
    results/preflight/hp_flatness/scale_coupling_summary.json
        per-arch: optima at each D, grid-step distance from locked
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "preflight"
FIG_DIR = RESULTS / "figures"


def _load_grid_for(d_train: int, arch: str) -> dict[tuple[float, int], float]:
    """Load val_mse per (lr, bs) for one (arch, D)."""
    if d_train == 600_000:
        base = RESULTS / "task3_lr_bs" / arch
    else:
        base = RESULTS / f"task3b_lr_bs_d{d_train}" / arch
    if not base.exists():
        return {}
    cells: dict[tuple[float, int], float] = {}
    for f in base.rglob("result.json"):
        d = json.loads(f.read_text())
        cell = f.parts[-3]
        try:
            lr_str, bs_str = cell.split("_")
            lr = float(lr_str.lstrip("lr"))
            bs = int(bs_str.lstrip("bs"))
        except ValueError:
            continue
        cells[(lr, bs)] = float(d.get("best_val_mse", float("inf")))
    return cells


def main():
    decisions_path = RESULTS / "pre_flight_decisions.yaml"
    decisions = yaml.safe_load(decisions_path.read_text()) if decisions_path.exists() else {}

    archs = ("legnet", "dream_rnn", "dream_attn")
    Ds = (500, 6000, 60000, 600000)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    coupling_summary: dict[str, dict] = {}

    for arch in archs:
        cells_by_d: dict[int, dict[tuple[float, int], float]] = {
            d: _load_grid_for(d, arch) for d in Ds
        }
        if not any(cells_by_d.values()):
            print(f"  [{arch}] no data yet — skip")
            continue

        # Build a single shared LR×BS axis pulling all unique values across all D
        all_lrs = sorted({lr for cells in cells_by_d.values() for (lr, _) in cells})
        all_bss = sorted({bs for cells in cells_by_d.values() for (_, bs) in cells})

        fig, axes = plt.subplots(1, len(Ds), figsize=(5 * len(Ds), 4.5), sharey=True)
        if len(Ds) == 1:
            axes = [axes]
        per_d_optima: dict[int, dict] = {}
        for ax, d in zip(axes, Ds):
            cells = cells_by_d[d]
            M = np.full((len(all_lrs), len(all_bss)), np.nan)
            for i, lr in enumerate(all_lrs):
                for j, bs in enumerate(all_bss):
                    if (lr, bs) in cells:
                        M[i, j] = cells[(lr, bs)]
            im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r")
            # Per-D optimum (lowest val_mse cell)
            if not np.isnan(M).all():
                opt_idx = np.unravel_index(int(np.nanargmin(M)), M.shape)
                opt_lr = all_lrs[opt_idx[0]]
                opt_bs = all_bss[opt_idx[1]]
                opt_val = float(M[opt_idx])
                per_d_optima[d] = {
                    "lr": opt_lr, "bs": opt_bs, "val_mse": opt_val,
                    "n_cells": int(np.sum(~np.isnan(M))),
                }
                ax.scatter(
                    [opt_idx[1]], [opt_idx[0]],
                    marker="*", s=300, c="red", edgecolors="white", linewidths=1.5,
                    zorder=10, label=f"D={d}: lr={opt_lr:g}, bs={opt_bs}",
                )
            # Locked optimum (from YAML; only marked at D=600k panel)
            locked_lr = decisions.get("learning_rate", {}).get(arch, {}).get("value")
            locked_bs = decisions.get("batch_size", {}).get(arch, {}).get("value")
            if locked_lr is not None and locked_bs is not None:
                try:
                    li = all_lrs.index(float(locked_lr))
                    lj = all_bss.index(int(locked_bs))
                    ax.scatter(
                        [lj], [li],
                        marker="X", s=200, facecolors="none", edgecolors="cyan", linewidths=2,
                        zorder=11,
                    )
                except (ValueError, IndexError):
                    pass
            ax.set_xticks(range(len(all_bss)))
            ax.set_yticks(range(len(all_lrs)))
            ax.set_xticklabels([str(b) for b in all_bss])
            ax.set_yticklabels([f"{lr:.0e}" for lr in all_lrs])
            ax.set_xlabel("Batch size")
            ax.set_title(f"{arch} @ D={d}")
            ax.legend(fontsize=7, loc="upper right")
            plt.colorbar(im, ax=ax, label="val MSE")
        axes[0].set_ylabel("Learning rate")
        fig.suptitle(
            f"{arch}: LR × BS grid across D values "
            "(★ = per-D optimum; cyan ◇ = locked from D=600k)"
        )
        fig.tight_layout()
        out = FIG_DIR / f"task3b_heatmap_{arch}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out}")

        # Scale-coupling check: distance (in grid steps) between per-D optima
        if per_d_optima:
            ds_with_opt = sorted(per_d_optima.keys())
            anchor_d = max(ds_with_opt)  # D=600k is the locked anchor
            anchor = per_d_optima[anchor_d]
            distances = {}
            for d in ds_with_opt:
                opt = per_d_optima[d]
                lr_step = abs(all_lrs.index(opt["lr"]) - all_lrs.index(anchor["lr"]))
                bs_step = abs(all_bss.index(opt["bs"]) - all_bss.index(anchor["bs"]))
                distances[d] = {"lr_steps": lr_step, "bs_steps": bs_step}
            coupling_summary[arch] = {
                "per_d_optima": per_d_optima,
                "grid_step_distances_from_locked": distances,
                "max_lr_step_drift": max(d["lr_steps"] for d in distances.values()),
                "max_bs_step_drift": max(d["bs_steps"] for d in distances.values()),
            }
            mlr = coupling_summary[arch]["max_lr_step_drift"]
            mbs = coupling_summary[arch]["max_bs_step_drift"]
            verdict = "stable" if (mlr <= 1 and mbs <= 1) else "scale_coupling_flagged"
            coupling_summary[arch]["verdict"] = verdict
            print(f"  [{arch}] max LR drift {mlr} steps, max BS drift {mbs} steps → {verdict}")

    out_dir = RESULTS / "hp_flatness"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scale_coupling_summary.json").write_text(json.dumps(coupling_summary, indent=2))
    print(f"\nSaved {out_dir / 'scale_coupling_summary.json'}")


if __name__ == "__main__":
    main()
