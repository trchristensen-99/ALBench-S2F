"""Two figures for the results that have no picture yet.

fig18  THE REALISED MIXTURE. When a rule selects from a union of reservoirs it is choosing
       the mixture, and the composition of what it chose is a direct, data-driven answer to
       "which reservoirs, in what ratio" -- the question the 4M order actually asks. Two
       independent measurements of it: the 7-way union behind the ab54 genomic-pilot arms,
       and the 4-way union of the composition sweep, where three different rules chose from
       the same pool. Random is the control: it must come out uniform, and it does.

fig19  THE WINNER IS ENDPOINT-RELATIVE. Each eval set is won by the candidate pool whose
       sequences resemble it, and the ordering barely moves when the rule changes. There is
       no universal best corpus, which is the argument for reporting several independent
       endpoints rather than one.

    python scratchpad/nov25/mixture_figures.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "orchestration"))
sys.path.insert(0, str(REPO / "scripts/analysis"))
from read_acq52 import read_run  # noqa: E402

A52 = REPO / "outputs/acq52"
POOLS = A52 / "pools"
BLUE, VERM, GREEN, PURPLE, ORANGE, SKY = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
GREY = "#6b6b6b"

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)

RULE_COLOUR = {
    "uncertainty": BLUE,
    "random": GREY,
    "expected_activity": VERM,
}
RULE_LABEL = {
    "uncertainty": "uncertainty",
    "random": "random (control)",
    "expected_activity": "expected activity",
}


def mix(name: str) -> dict | None:
    p = POOLS / f"{name}_mix.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _share_panel(ax, blobs: list[tuple[str, dict]], title: str) -> None:
    """Grouped bars of realised share, against the uniform share of the same pool."""
    names = sorted({k for _, b in blobs for k in b["realised_mixture"]})
    # Order by what the first rule chose, so the preference is readable down the axis.
    first = blobs[0][1]["realised_mixture"]
    names.sort(key=lambda n: -first.get(n, 0))
    uniform = 100.0 / len(names)
    h = 0.8 / len(blobs)
    y = np.arange(len(names))
    for i, (rule, b) in enumerate(blobs):
        tot = sum(b["realised_mixture"].values())
        vals = [100.0 * b["realised_mixture"].get(n, 0) / tot for n in names]
        ax.barh(
            y + (i - (len(blobs) - 1) / 2) * h,
            vals,
            height=h * 0.92,
            color=RULE_COLOUR.get(rule, GREY),
            edgecolor="black",
            linewidth=0.3,
            label=f"{RULE_LABEL.get(rule, rule)}  (selected activity {b['label_mean']:.2f})",
        )
        for yy, v in zip(y + (i - (len(blobs) - 1) / 2) * h, vals, strict=True):
            ax.text(v + 0.35, yy, f"{v:.1f}", va="center", fontsize=7, color=GREY)
    ax.axvline(uniform, color="black", ls="--", lw=1.1)
    ax.text(
        uniform,
        len(names) - 0.35,
        f"  equal shares ({uniform:.1f}%)",
        fontsize=8,
        color="black",
        va="top",
    )
    ax.set_yticks(y, names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("share of the selected corpus (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")


def fig18(out: Path) -> None:
    seven = [(r, mix(f"gpilot_{r}_d105000")) for r in ("uncertainty", "random")]
    seven = [(r, b) for r, b in seven if b]
    four = [
        (r, mix(f"pool{p}_14_d30000"))
        for r, p in (("uncertainty", "B"), ("expected_activity", "E"), ("random", "R"))
    ]
    four = [(r, b) for r, b in four if b]
    if not seven and not four:
        print("fig18: no mixture sidecars found")
        return

    panels = [p for p in (seven, four) if p]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.9 * len(panels), 4.6), squeeze=False)
    titles = []
    if seven:
        n = seven[0][1]["n_candidates"]
        titles.append(
            f"(a) 7-way union, {n // 1000}k scored, 75k selected onto a 30k genomic pilot"
        )
    if four:
        n = four[0][1]["n_candidates"]
        titles.append(f"(b) 4-way union, {n // 1000}k scored, 30k selected, no pilot")
    for ax, blobs, t in zip(axes[0], panels, titles, strict=True):
        _share_panel(ax, blobs, t)
    fig.suptitle(
        "What the rule chose when it could choose the mixture",
        y=1.03,
        x=0.02,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "fig18_realised_mixture.png")
    plt.close(fig)
    print(f"fig18 written ({len(seven)} + {len(four)} arms)")
    for tag, blobs in (("7-way", seven), ("4-way", four)):
        for rule, b in blobs:
            tot = sum(b["realised_mixture"].values())
            sh = {
                k: round(100 * v / tot, 1)
                for k, v in sorted(b["realised_mixture"].items(), key=lambda kv: -kv[1])
            }
            print(f"  {tag} {rule:<18} activity {b['label_mean']:+.3f}  {sh}")


# fig19 -- the single-reservoir arms of the composition sweep. Index order matches the
# sbatch arrays: 00 random, 01 motif_shared_core, 02 evoaug, 03 genomic.
SINGLES = [
    (0, "random", GREEN),
    (1, "motif_shared_core", VERM),
    (2, "evoaug", PURPLE),
    (3, "genomic", BLUE),
]
RULES19 = [("poolB", "uncertainty"), ("poolR", "random"), ("poolE", "expected activity")]
ENDPOINTS = [
    ("ood", "high-activity designed"),
    ("genomic", "genomic"),
    ("random_32k", "random"),
]


def _arm(name: str) -> dict[str, float] | None:
    d = A52 / name
    if not d.is_dir():
        return None
    kids = [k for k in sorted(d.iterdir()) if k.is_dir()]
    got = read_run(kids[0] if len(kids) == 1 else d)
    return None if got is None else got[1]


def fig19(out: Path) -> None:
    have: dict[tuple[str, int], dict[str, float]] = {}
    for pfx, rule in RULES19:
        for j, _name, _c in SINGLES:
            st = _arm(f"{pfx}_{j:02d}")
            if st:
                have[(rule, j)] = st
    if not have:
        print("fig19: no single-reservoir arms finished")
        return
    rules = [r for _, r in RULES19 if any(k[0] == r for k in have)]

    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(4.7 * len(ENDPOINTS), 4.3))
    w = 0.8 / max(1, len(rules))
    for ax, (key, lab) in zip(np.atleast_1d(axes), ENDPOINTS, strict=True):
        x = np.arange(len(SINGLES))
        vals_all = []
        for i, rule in enumerate(rules):
            vals = [have.get((rule, j), {}).get(key, np.nan) for j, _n, _c in SINGLES]
            vals_all += [v for v in vals if np.isfinite(v)]
            ax.bar(
                x + (i - (len(rules) - 1) / 2) * w,
                vals,
                width=w * 0.9,
                color=RULE_COLOUR.get(rule.replace(" ", "_"), GREY),
                edgecolor="black",
                linewidth=0.3,
                label=rule,
            )
            good = [(j, v) for j, v in enumerate(vals) if np.isfinite(v)]
            if good:
                jb = max(good, key=lambda t: t[1])[0]
                ax.plot(
                    [x[jb] + (i - (len(rules) - 1) / 2) * w],
                    [max(good, key=lambda t: t[1])[1]],
                    marker="*",
                    ms=11,
                    color="black",
                    ls="",
                    zorder=5,
                )
        if vals_all:
            lo, hi = min(vals_all), max(vals_all)
            ax.set_ylim(lo - 0.12 * (hi - lo + 0.01), hi + 0.06 * (hi - lo + 0.01))
        ax.set_xticks(
            x,
            [n.replace("motif_shared_core", "motif") for _j, n, _c in SINGLES],
            rotation=20,
            ha="right",
        )
        ax.set_title(f"eval set: {lab}")
        ax.set_xlabel("the one reservoir the rule chose from")
    np.atleast_1d(axes)[0].set_ylabel("stacked r   (D = 30,000)")
    np.atleast_1d(axes)[0].legend(
        title="acquisition rule", title_fontsize=9, fontsize=8.5, loc="lower left"
    )
    fig.suptitle(
        "No universal best corpus: each endpoint is won by the pool that resembles it "
        "(★ = best, within rule)",
        y=1.02,
        x=0.02,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "fig19_endpoint_relative_winner.png")
    plt.close(fig)
    print(f"fig19 written ({len(have)} arms)")
    for key, lab in ENDPOINTS:
        row = []
        for rule in rules:
            vals = [(n, have.get((rule, j), {}).get(key, np.nan)) for j, n, _c in SINGLES]
            vals = [(n, v) for n, v in vals if np.isfinite(v)]
            if vals:
                b = max(vals, key=lambda t: t[1])
                row.append(f"{rule}: {b[0]} {b[1]:.3f}")
        print(f"  {lab:<24} " + " | ".join(row))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/nov25")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    fig18(out)
    fig19(out)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
