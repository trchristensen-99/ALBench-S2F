"""Figure 2 in ERROR space, derived from the Pearson r we already have (no re-run needed).

Two transforms, because they answer different questions:
  (1-r)^2   - as requested; a monotone error-like transform of r
  1 - r^2   - the variance-normalised MSE (nMSE). This is the quantity that equals MSE/Var(y) for an
              optimally scaled predictor, so it is the principled bridge to a real MSE curve and is
              what the measured MSE plot will approximate once those runs land.
Log-log panels are included because a power law error ~ D^-alpha is a STRAIGHT LINE there, so the
scaling exponent is read directly off the slope -- which is the main reason to want error space.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.expanduser("~/Downloads/pi_meeting_aug19")
D = np.array([3000, 10000, 30000, 100000, 300000])
R = {
    "evoaug_heavy": (
        [0.7747, 0.8197, 0.8556, 0.8910, 0.9135],
        [0.4402, 0.4498, 0.4841, 0.5490, 0.6188],
    ),
    "phylogenetic_zoonomia": (
        [0.7783, 0.8294, 0.8584, 0.8924, 0.9123],
        [0.4306, 0.4705, 0.5090, 0.5569, 0.6046],
    ),
    "motif_planted_v2": (
        [0.7534, 0.8065, 0.8492, 0.8846, 0.9068],
        [0.4868, 0.5292, 0.5831, 0.6392, 0.7152],
    ),
    "genomic": ([0.7684, 0.8082, 0.8432, 0.8755, 0.9023], [0.4180, 0.4812, 0.4893, 0.5405, 0.6049]),
    "dinuc_shuffle": (
        [0.6887, 0.7537, 0.8334, 0.8567, 0.8786],
        [0.4187, 0.4387, 0.4662, 0.5202, 0.6021],
    ),
    "random": ([0.6720, 0.7741, 0.8087, 0.8495, 0.8738], [0.4454, 0.5144, 0.5314, 0.6155, 0.6704]),
}
cols = plt.cm.tab10(np.linspace(0, 1, 10))
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
for i, (name, (g, o)) in enumerate(R.items()):
    g, o = np.array(g), np.array(o)
    # row 0 = in-distribution, row 1 = OOD
    for row, v in ((0, g), (1, o)):
        axes[row, 0].plot(
            D, (1 - v) ** 2, "o-", color=cols[i], lw=1.7, ms=4.5, label=name if row == 0 else None
        )
        axes[row, 1].plot(D, 1 - v**2, "o-", color=cols[i], lw=1.7, ms=4.5)
        axes[row, 2].plot(D, 1 - v**2, "o-", color=cols[i], lw=1.7, ms=4.5)

titles = [
    ("A. (1-r)^2   [log-linear]", "(1 - r)$^2$", "log", "linear"),
    ("B. 1-r^2 = nMSE   [log-linear]", "1 - r$^2$", "log", "linear"),
    ("C. 1-r^2   [LOG-LOG: power law = straight line]", "1 - r$^2$", "log", "log"),
]
for col, (ttl, yl, xs, ys) in enumerate(titles):
    for row, tag in ((0, "in-distribution"), (1, "OOD")):
        ax = axes[row, col]
        ax.set_xscale(xs)
        ax.set_yscale(ys)
        ax.set_xlabel("D (oracle-labelled sequences)")
        ax.set_ylabel(yl)
        ax.set_title(f"{ttl}\n{tag}", fontsize=9.5)
        ax.grid(alpha=0.3, which="both")
axes[0, 0].legend(fontsize=7.5, loc="upper right")

# fit and print power-law slopes on the log-log panel
for row, key in ((0, 0), (1, 1)):
    txt = []
    for i, (name, vals) in enumerate(R.items()):
        v = np.array(vals[key])
        sl = np.polyfit(np.log10(D), np.log10(1 - v**2), 1)[0]
        txt.append(f"{name[:14]}: {sl:+.3f}")
    axes[row, 2].text(
        0.02,
        0.03,
        "power-law slope (1-r²):\n" + "\n".join(txt),
        transform=axes[row, 2].transAxes,
        fontsize=6.4,
        va="bottom",
        bbox=dict(fc="white", alpha=0.8, ec="0.7"),
    )

fig.suptitle(
    "Scaling in ERROR space (derived from Pearson r — measured MSE curves pending)",
    fontsize=13,
    weight="bold",
)
fig.text(
    0.5,
    -0.015,
    "(1-r)² is the requested transform; 1-r² = nMSE is the principled bridge to true MSE (it equals MSE/Var(y) for an optimally scaled predictor).\n"
    "Log-log (C) is where a power law error ~ D^-α is a straight line, so α is read off the slope. NOTE these are DERIVED from r, not measured MSE:\n"
    "they cannot capture calibration/scale error, which is exactly what raw MSE adds. Treat slopes as indicative until the measured MSE grid lands.",
    ha="center",
    fontsize=8.4,
    color="0.3",
)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2c_error_curves.png", dpi=300, bbox_inches="tight")
print("wrote fig2c_error_curves.png")
