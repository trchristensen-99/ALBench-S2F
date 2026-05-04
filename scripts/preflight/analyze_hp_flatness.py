"""HP-flatness diagnostics over the joint LR×BS sweep (Task 3).

Per PI's meeting note: "evaluate flatness of HP search effects". This script
asks: across the LR×BS grid, how sensitive is best_val_mse to small HP
perturbations? A flat region around the optimum means we can lock HPs and
the main sweep is robust; a sharp region means small mistakes in HP transfer
will inflate γ_k variance.

Outputs (results/preflight/hp_flatness/):
    flatness_summary.json
        per-arch:
          - val_mse_at_optimum
          - 1st-ring relative range:  max-min over the 8 HP cells one grid
                                       step from the optimum, divided by
                                       optimum value (lower = flatter)
          - 2nd-ring relative range:  same over the 16 cells two grid steps
                                       away
    flatness_heatmap_<arch>.{png,pdf}: LR × BS heatmap of best_val_mse with
        the optimum cell highlighted.

Diagnostic interpretation:
    1st-ring rel-range < 0.05 → very flat near optimum → safe to lock
    1st-ring rel-range 0.05-0.15 → moderate sensitivity → verify at D_min
    1st-ring rel-range > 0.15 → sharp; inspect the heatmap before locking

Usage:
    uv run --no-sync python scripts/preflight/analyze_hp_flatness.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    base = REPO / "results" / "preflight" / "task3_lr_bs"
    if not base.exists():
        print(f"No Task 3 results yet at {base}")
        return
    out_dir = REPO / "results" / "preflight" / "hp_flatness"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Walk: results/preflight/task3_lr_bs/<arch>/lr<lr>_bs<bs>/seed<seed>/result.json
    by_arch: dict[str, dict[tuple[float, int], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for f in sorted(base.rglob("result.json")):
        d = json.loads(f.read_text())
        parts = f.parts
        arch = parts[-4]
        cell = parts[-3]  # "lr1e-3_bs256"
        lr_str, bs_str = cell.split("_")
        lr = float(lr_str.lstrip("lr"))
        bs = int(bs_str.lstrip("bs"))
        by_arch[arch][(lr, bs)].append(float(d.get("best_val_mse", 0)))

    summary = {}
    for arch, grid in by_arch.items():
        if not grid:
            continue
        # Build LR × BS matrix (mean across seeds at each cell)
        lrs = sorted({lr for (lr, _) in grid})
        bss = sorted({bs for (_, bs) in grid})
        M = np.full((len(lrs), len(bss)), np.nan)
        for i, lr in enumerate(lrs):
            for j, bs in enumerate(bss):
                vals = grid.get((lr, bs))
                if vals:
                    M[i, j] = float(np.mean(vals))

        if np.isnan(M).all():
            continue
        # Find optimum cell
        flat = np.nan_to_num(M, nan=np.inf)
        opt_idx = np.unravel_index(int(np.argmin(flat)), M.shape)
        opt_val = float(M[opt_idx])

        # 1st-ring: 8 cells (i±1, j±1) excluding center
        # 2nd-ring: 16 cells two steps away
        def ring_vals(M, opt_idx, k):
            i0, j0 = opt_idx
            vs = []
            for di in range(-k, k + 1):
                for dj in range(-k, k + 1):
                    if max(abs(di), abs(dj)) != k:
                        continue
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < M.shape[0] and 0 <= j < M.shape[1] and not np.isnan(M[i, j]):
                        vs.append(float(M[i, j]))
            return vs

        ring1 = ring_vals(M, opt_idx, 1)
        ring2 = ring_vals(M, opt_idx, 2)
        rel_range_1 = (max(ring1) - min(ring1)) / opt_val if ring1 else None
        rel_range_2 = (max(ring2) - min(ring2)) / opt_val if ring2 else None

        # Heatmap
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r")
        ax.set_xticks(range(len(bss)))
        ax.set_yticks(range(len(lrs)))
        ax.set_xticklabels([str(b) for b in bss])
        ax.set_yticklabels([f"{lr:.0e}" for lr in lrs])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Learning rate")
        ax.set_title(f"{arch} — best val MSE @ D=600k")
        # Highlight optimum
        ax.scatter([opt_idx[1]], [opt_idx[0]], marker="*", s=300, c="red", edgecolors="black")
        plt.colorbar(im, ax=ax, label="val MSE (lower = better)")
        fig.tight_layout()
        fig.savefig(out_dir / f"flatness_heatmap_{arch}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out_dir / f"flatness_heatmap_{arch}.pdf", bbox_inches="tight")
        plt.close(fig)

        summary[arch] = {
            "optimum": {
                "lr": lrs[opt_idx[0]],
                "batch_size": bss[opt_idx[1]],
                "val_mse": opt_val,
            },
            "1st_ring_rel_range": rel_range_1,
            "2nd_ring_rel_range": rel_range_2,
            "n_cells": int(np.sum(~np.isnan(M))),
            "interpretation": (
                "very flat"
                if (rel_range_1 or 1) < 0.05
                else "moderate"
                if (rel_range_1 or 1) < 0.15
                else "sharp"
            ),
        }
        print(f"\n=== {arch} ===")
        print(json.dumps(summary[arch], indent=2))

    out = out_dir / "flatness_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
