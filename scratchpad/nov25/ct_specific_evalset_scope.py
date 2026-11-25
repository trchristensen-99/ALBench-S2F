"""How big, and how precise, can a cell-type-SPECIFIC eval set be?

A specificity eval set is only worth building if the specificity it selects on is measured
well enough to score against. The obvious construction -- take the top decile of
(target - others) -- is a trap: the Gosai difference carries the noise of both measurements,
so the extreme tail of a noisy difference is enriched for noisy ROWS, not for specific
sequences. That is the regression-to-the-extremes failure, and the resulting set would have
a measurement ceiling near zero while looking like the hardest set in the panel.

This measures the trade-off directly on the held-out test chromosomes (chr7/chr13), for
every candidate selection rule, and reports for each:

    n                how many sequences survive
    ceiling          sqrt(1 - E[SE^2]/Var) on the DIFFERENCE -- the most any model can score
    mean |delta|     how specific the selected sequences actually are
    frac active      how many are also active in the target cell, which is what a
                     ct-specific DESIGN campaign needs

Run on the login node: it reads one table and does no GPU work.

    python scratchpad/nov25/ct_specific_evalset_scope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from albench.celltype import data as D  # noqa: E402

TEST_CHROMS = ("chr7", "chr13")


def ceiling(y: np.ndarray, se: np.ndarray) -> float:
    var, noise = float(np.var(y)), float(np.mean(np.square(se)))
    return float(np.sqrt(max(0.0, 1 - noise / var))) if var > 0 else float("nan")


def main() -> int:
    d = D.load_gosai()
    cells = list(d.cells)
    # MultiCellData.split() asserts the chromosome holdout actually matched, which is the
    # guard that exists because Gosai writes chromosomes as bare "7"/"13".
    test = d.split()["test"]
    print(f"{len(cells)} cells {cells}; {int(test.sum()):,} test rows")

    for target in cells:
        j = cells.index(target)
        others = [k for k in range(len(cells)) if k != j]
        # Measured in the target AND in every other cell: the difference is only defined
        # where both sides exist.
        m = test & d.mask[:, j]
        for o in others:
            m &= d.mask[:, o]
        sel = np.flatnonzero(m)
        yt = d.y[sel, j]
        yo = d.y[sel][:, others].max(axis=1)  # vs the STRONGEST competitor, not the mean
        st = d.se[sel, j]
        so = d.se[sel][:, others].max(axis=1)
        dy = yt - yo
        dse = np.sqrt(st**2 + so**2)
        z = dy / np.maximum(dse, 1e-6)
        act_hi = yt >= np.quantile(d.y[test & d.mask[:, j], j], 0.90)

        print(f"\n=== target {target} | {len(sel):,} rows measured in all {len(cells)} cells")
        print(
            f"    difference: sd {dy.std():.3f}, mean SE {dse.mean():.3f}, "
            f"ceiling {ceiling(dy, dse):.3f}"
        )
        hdr = f"{'selection rule':<34}{'n':>8}{'ceiling':>9}{'mean |d|':>10}{'frac active':>13}"
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        rules: list[tuple[str, np.ndarray]] = [
            ("all measured (no selection)", np.ones(len(sel), bool)),
            ("top decile of raw delta", dy >= np.quantile(dy, 0.90)),
            ("top 5% of raw delta", dy >= np.quantile(dy, 0.95)),
            ("|z| >= 2 (precision-gated)", np.abs(z) >= 2),
            ("z >= 2 (target-specific UP)", z >= 2),
            ("z >= 3", z >= 3),
            ("z >= 2 AND active in target", (z >= 2) & act_hi),
            ("z <= -2 (specific DOWN)", z <= -2),
        ]
        for name, k in rules:
            if k.sum() < 50:
                print(f"    {name:<34}{int(k.sum()):>8}  (too few)")
                continue
            print(
                f"    {name:<34}{int(k.sum()):>8}{ceiling(dy[k], dse[k]):>9.3f}"
                f"{np.abs(dy[k]).mean():>10.3f}{act_hi[k].mean():>13.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
