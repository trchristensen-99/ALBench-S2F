"""Figure 2 variant: MSE loss curves instead of Pearson r.

IMPORTANT interpretive caveat, printed on the figure:
r is invariant to scale and shift; MSE is not. A reservoir whose oracle labels have SMALLER variance
will show lower MSE without the model being any better -- predicting a narrower activity distribution
is easier in squared error. So we plot raw MSE (panels A/B) AND variance-normalised MSE, nMSE = MSE /
Var(y) = 1 - R^2 (panels C/D). Only the normalised version is comparable ACROSS reservoirs; the raw
version is comparable across D within one reservoir.
"""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "outputs/fm_packed"
OUT = os.path.expanduser("~/Downloads/pi_meeting_aug19")
os.makedirs(OUT, exist_ok=True)

rows = {}
for f in glob.glob(os.path.join(ROOT, "mse_*", "fm_scaling_point.json")):
    d = json.load(open(f))
    name = os.path.basename(os.path.dirname(f))  # mse_<reservoir>_d<D>
    res = "_".join(name.split("_")[1:-1])
    D = int(name.split("_d")[-1])
    m = d.get("metrics", {})
    if "genomic__mse" not in m:
        continue
    rows.setdefault(res, []).append(
        (
            D,
            m["genomic__mse"],
            m.get("ood__mse", np.nan),
            m.get("genomic__nmse", np.nan),
            m.get("ood__nmse", np.nan),
        )
    )
if not rows:
    print(f"no MSE-bearing runs yet under {ROOT}/mse_* -- rerun once the grid finishes")
    sys.exit(0)

fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
cols = plt.cm.tab10(np.linspace(0, 1, 10))
for i, (res, vals) in enumerate(sorted(rows.items())):
    v = np.array(sorted(vals))
    D = v[:, 0]
    for ax, col, lab in (
        (axes[0, 0], 1, None),
        (axes[0, 1], 2, None),
        (axes[1, 0], 3, None),
        (axes[1, 1], 4, None),
    ):
        ax.plot(
            D,
            v[:, col],
            "o-",
            color=cols[i],
            label=res if ax is axes[0, 0] else lab,
            lw=1.7,
            ms=4.5,
        )
for ax, ttl, yl in zip(
    axes.ravel(),
    [
        "A. in-distribution — raw MSE",
        "B. OOD — raw MSE",
        "C. in-distribution — nMSE = 1 - R²",
        "D. OOD — nMSE = 1 - R²",
    ],
    ["MSE", "MSE", "MSE / Var(y)", "MSE / Var(y)"],
):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("D (oracle-labelled training sequences)")
    ax.set_ylabel(yl)
    ax.set_title(ttl, fontsize=10.5)
    ax.grid(alpha=0.3, which="both")
axes[0, 0].legend(fontsize=8, loc="upper right")
fig.suptitle(
    "Scaling in LOSS space — MSE and variance-normalised MSE", fontsize=12.5, weight="bold"
)
fig.text(
    0.5,
    -0.015,
    "Raw MSE (A,B) is comparable across D WITHIN a reservoir but NOT between reservoirs: a strategy whose oracle labels have\n"
    "smaller variance gets lower MSE for free. nMSE = MSE/Var(y) = 1 - R² (C,D) removes that and is the cross-reservoir comparison.\n"
    "Log-log axes make a power law appear as a straight line, so the slope is directly readable.",
    ha="center",
    fontsize=8.6,
    color="0.3",
)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2b_reservoir_scaling_MSE.png", dpi=300, bbox_inches="tight")
print(
    f"wrote fig2b_reservoir_scaling_MSE.png from {sum(len(v) for v in rows.values())} cells "
    f"across {len(rows)} reservoirs"
)
