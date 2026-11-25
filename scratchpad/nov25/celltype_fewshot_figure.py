"""What one cell type buys you in another, and what the second cell's data then adds.

The order measures two cell lines. That makes one question sharper than the seven-cell
conditioning work ever made it: if you already have a corpus measured in HepG2, how many
K562 measurements do you still need -- and is that fewer than starting from nothing?

`albench.celltype.fewshot` answers it as a curve on two axes, pretrained cell types
(0, 1, 3, 6; 0 is from scratch) against labelled sequences in the target cell. The
from-scratch row at the SAME number of target sequences is the only honest comparison:
more data always helps, so a curve that only beat its own zero-shot point would prove
nothing.

fs41-43 only ever targeted a third cell (SK-N-SH, GM12878, Jurkat). fs56 adds the pair the
collaborators can actually run -- K562 from HepG2 and HepG2 from K562.

    python scratchpad/nov25/celltype_fewshot_figure.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BLUE, VERM, GREEN, PURPLE, ORANGE, SKY = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
)
GREY = "#6b6b6b"
# Colour by how many cell types the model was pretrained on, never by target, so the same
# colour means the same thing in every panel.
PRE_COLOUR = {0: GREY, 1: VERM, 3: BLUE, 6: GREEN}
# The pretrain pool is CELL_ORDER minus the target, so "1 cell" is HepG2 for every target
# except HepG2 itself, where it is K562.
PRE_LABEL = {0: "from scratch", 1: "pretrained on 1 cell", 3: "on 3 cells", 6: "on 6 cells"}

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


def load(roots: list[Path]) -> dict[tuple[str, str], dict]:
    """(target, mode) -> {pretrain: {shots: r}, 'ceiling': c, 'pre_cells': {...}}."""
    out: dict[tuple[str, str], dict] = {}
    for root in roots:
        for f in sorted(root.glob("*/curve.json")):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            rows = d.get("results", [])
            # dp48 writes a curve.json too, with the same filename and a different schema
            # (strategy/shots and r_all rather than pretrain_cells/r). Without this check it
            # contributes an empty record and a panel with no lines in it.
            rows = [r for r in rows if "pretrain_cells" in r and "r" in r]
            if len(rows) < 4:
                continue  # a run that is still going, or the smoke test
            key = (d["target"], d["args"].get("mode", "?"))
            rec = out.setdefault(
                key, {"curves": {}, "ceiling": float("nan"), "pre_cells": {}, "src": f.parent.name}
            )
            for r in rows:
                v = r.get("r")
                if not isinstance(v, (int, float)) or not math.isfinite(v):
                    continue
                rec["curves"].setdefault(int(r["pretrain_cells"]), {})[int(r["shots"])] = float(v)
                rec["pre_cells"][int(r["pretrain_cells"])] = list(r.get("pretrain", []))
                if math.isfinite(r.get("ceiling", float("nan"))):
                    rec["ceiling"] = float(r["ceiling"])
    return out


def equivalent_shots(scratch: dict[int, float], target_r: float) -> float | None:
    """How many from-scratch sequences reach `target_r`, log-interpolated between rungs.

    This is the number that answers the budget question -- "one cell type already measured
    is worth N measurements in this one" -- and it is only defined inside the measured
    range, so a value past the last rung is reported as None rather than extrapolated.
    """
    pts = sorted((k, v) for k, v in scratch.items() if k > 0)
    if not pts or target_r <= pts[0][1]:
        return float(pts[0][0]) if pts else None
    for (k0, v0), (k1, v1) in zip(pts[:-1], pts[1:], strict=False):
        if v0 <= target_r <= v1:
            if v1 == v0:
                return float(k1)
            f = (target_r - v0) / (v1 - v0)
            return float(10 ** (math.log10(k0) + f * (math.log10(k1) - math.log10(k0))))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument(
        "--roots",
        nargs="+",
        default=["outputs/celltype_curves", "outputs/celltype", "outputs/fs56"],
    )
    ap.add_argument("--mode", default="", help="restrict to one conditioning mode")
    ap.add_argument("--only", nargs="*", default=[], help="restrict to these target cells")
    ap.add_argument("--name", default="fig20_celltype_fewshot.png")
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    data = load([REPO / r if not Path(r).is_absolute() else Path(r) for r in a.roots])
    if not data:
        print("no few-shot curves found")
        return 1

    # One curve per target: prefer a CONDITIONED model where one was run, because that is
    # what would actually be deployed on a two-cell library; fall back to the cell-blind
    # one, and say which in the panel title.
    # `agnostic` first, not last: it is the only mode every target was run in, and a figure
    # whose panels each used a different conditioning mode would compare two things at once.
    order = ["agnostic", "onehot", "text", "text_shuffled"]
    by_target: dict[str, tuple[str, dict]] = {}
    for (tgt, mode), rec in data.items():
        if a.mode and mode != a.mode:
            continue
        if a.only and tgt not in a.only:
            continue
        cur = by_target.get(tgt)
        if cur is None or order.index(mode) < order.index(cur[0]):
            by_target[tgt] = (mode, rec)
    # The two cells the library will be measured in come first.
    tgts = sorted(by_target, key=lambda t: (t not in ("K562", "HepG2"), t))
    print(f"targets: {[(t, by_target[t][0]) for t in tgts]}")

    fig, axes = plt.subplots(
        2, len(tgts), figsize=(3.7 * len(tgts), 7.4), squeeze=False, sharex=True
    )
    for col, tgt in enumerate(tgts):
        mode, rec = by_target[tgt]
        scratch = rec["curves"].get(0, {})
        ax = axes[0][col]
        for npre in sorted(rec["curves"]):
            pts = sorted((k, v) for k, v in rec["curves"][npre].items())
            xs = [max(k, 30) for k, _ in pts]  # 0 shots drawn at the left edge of a log axis
            ys = [v for _, v in pts]
            cells = rec["pre_cells"].get(npre) or []
            lab = PRE_LABEL.get(npre, f"on {npre} cells")
            if npre == 1 and cells:
                lab += f" ({cells[0]})"
            ax.plot(
                xs,
                ys,
                "o-" if npre else "o--",
                color=PRE_COLOUR.get(npre, GREY),
                ms=4.5,
                lw=2.2 if npre == 1 else 1.7,
                label=lab,
            )
        if math.isfinite(rec["ceiling"]):
            ax.axhline(rec["ceiling"], color="black", ls=":", lw=1.1)
            # Several pretrained curves sit ABOVE this line. The ceiling is
            # sqrt(1 - E[SE^2]/Var) from the replicate error of that cell's own
            # measurements, so exceeding it means the error term is overstated for this
            # cell, not that the model beat the data. Said on the figure rather than hidden.
            top = max(
                (v for c in rec["curves"].values() for v in c.values()), default=rec["ceiling"]
            )
            note = f" measurement ceiling {rec['ceiling']:.2f}"
            if top > rec["ceiling"] + 0.01:
                note += "  (curves above it: its SE term is overstated)"
            ax.text(32, rec["ceiling"], note, fontsize=7, va="bottom")
        ax.set_xscale("log")
        ax.set_title(f"target {tgt}  ({mode})")
        if col == 0:
            ax.set_ylabel("r on the target cell, test chromosomes")

        # --- what the pretraining is worth, in sequences -----------------------
        ax = axes[1][col]
        rows = []
        kmax = max([k for k in scratch if k > 0], default=0)
        for npre in sorted(k for k in rec["curves"] if k > 0):
            for k, v in sorted(rec["curves"][npre].items()):
                if k == 0:
                    continue
                eq = equivalent_shots(scratch, v)
                # A pretrained model better than from-scratch at its LARGEST measured size
                # has no equivalent inside the measured range. That is a lower bound, drawn
                # hollow, not a missing point -- Jurkat is entirely in this regime.
                bound = eq is None
                rows.append((npre, max(k, 30), (kmax if bound else eq) / max(k, 30), bound))
        for npre in sorted({r[0] for r in rows}):
            pts = [(k, g, b) for n, k, g, b in rows if n == npre]
            colour = PRE_COLOUR.get(npre, GREY)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                "-",
                color=colour,
                lw=2.2 if npre == 1 else 1.7,
            )
            for x, g, b in pts:
                ax.plot(
                    [x],
                    [g],
                    marker="^" if b else "o",
                    ms=6 if b else 4.5,
                    mfc="white" if b else colour,
                    mec=colour,
                    mew=1.4,
                    ls="",
                )
        if any(p[3] for p in rows):
            ax.text(
                0.97,
                0.05,
                "open ▲ = lower bound\n(better than 100k from scratch)",
                transform=ax.transAxes,
                ha="right",
                fontsize=7,
                color=GREY,
            )
        ax.axhline(1.0, color="black", lw=1.0)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("sequences measured in the target cell")
        if col == 0:
            ax.set_ylabel("x more data a from-scratch model\nwould need for the same r")
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Panels differ in which pretrain levels they have; collect every label once.
    for col in range(1, len(tgts)):
        h2, l2 = axes[0][col].get_legend_handles_labels()
        for h, lab in zip(h2, l2, strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncols=min(4, len(labels)),
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "One cell type already measured, then sequences in the second",
        y=0.995,
        x=0.02,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out / a.name)
    plt.close(fig)
    print(f"{a.name} written")

    for tgt in tgts:
        mode, rec = by_target[tgt]
        scratch = rec["curves"].get(0, {})
        one = rec["curves"].get(1, {})
        if not scratch or not one:
            continue
        zs = one.get(0)
        line = f"  {tgt:<9} ({mode:<13}) zero-shot from 1 cell r={zs:.3f}" if zs else f"  {tgt}"
        for k in (1000, 10000, 100000):
            if k in one and k in scratch:
                eq = equivalent_shots(scratch, one[k])
                line += (
                    f" | +{k // 1000}k: {one[k]:.3f} vs {scratch[k]:.3f} scratch"
                    f" (= {eq / k:.1f}x data)"
                    if eq
                    else ""
                )
        print(line)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
