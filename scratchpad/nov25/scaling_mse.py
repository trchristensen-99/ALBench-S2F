"""Scaling curves in MSE, on every eval set, for the tuned per-size LegNet.

Pearson r compresses badly at the top of the range -- 0.92 and 0.93 look like nothing, and
the scaling behaviour is invisible. MSE on a log-log axis against training-set size is the
standard way to show it, and the slope is then a number worth quoting.

THE RECIPE MAP IS NOT OPTIONAL. The LegNet recipe is tuned per size (docs/TUNED_LEGNET.md):
a 1M-row corpus trained with the 30k recipe is badly overtrained and a 10k corpus trained
with the 1M recipe is undertrained, and either would put a training artefact into the curve
and call it scaling. Only the cell matching this map is read, and a size with no matching
cell is left out rather than substituted.

    10k -> e800_wd4   30k -> e500_wd2   100k -> e500   300k -> e300_wd05   1M -> e200_wd05

Endpoints are the 18 stored eval sets plus the 13 derived EFFECT endpoints
(pred_X - pred_genomic against oracle_X - oracle_genomic), the same set
eval_panel_selection.py uses.

    python scratchpad/nov25/scaling_mse.py --out outputs/figures/nov25
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "orchestration"))
from ens_util import enet_all_pos  # noqa: E402

ROOTS = [REPO / "outputs/wed41", REPO / "outputs/menu_retrain_v0"]
TUNED = {
    10_000: "e800_wd4",
    30_000: "e500_wd2",
    100_000: "e500",
    300_000: "e300_wd05",
    1_000_000: "e200_wd05",
}
RESERVOIRS = [
    ("genomic", ""),  # the bare cell name carries no suffix
    ("random", "_random"),
    ("motif_shared_core", "_motif_shared_core"),
    ("motif_ct_enriched", "_motif_ct_enriched"),
    ("evoaug", "_evoaug"),
    ("mutagenesis", "_mutagenesis"),
    ("gc_matched", "_gc_matched"),
    ("dinuc_shuffle", "_dinuc_shuffle"),
]
PERTURB = (
    "dinuc_shuffle",
    "sub_low",
    "sub_med",
    "sub_high",
    "ins_low",
    "ins_med",
    "ins_high",
    "del_low",
    "del_med",
    "del_high",
    "translocation",
    "inversion",
)
# Okabe-Ito
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]


def cell_dir(res_suffix: str, D: int) -> Path | None:
    name = f"bench38_ln_{TUNED[D]}_{D // 1000}k_c20{res_suffix}"
    for r in ROOTS:
        p = r / name
        if p.is_dir() and glob.glob(str(p / "*_meta.json")):
            return p
    return None


def endpoint_mse(d: Path) -> dict[str, float] | None:
    """Stacked-ensemble MSE on every level and effect endpoint for one trained cell."""
    lab_p = d / "labels.npz"
    metas = sorted(glob.glob(str(d / "*_meta.json")))
    if not metas or not lab_p.exists():
        return None
    lab = np.load(lab_p, allow_pickle=True)
    sets = [k[len("oracle_") :] for k in lab.files if k.startswith("oracle_")]
    Y = {s: np.asarray(lab[f"oracle_{s}"], float).ravel() for s in sets}
    yv = np.asarray(lab["val_labels"], float).ravel()

    V, T = [], {s: [] for s in sets}
    for m in metas:
        z = np.load(m.replace("_meta.json", ".npz"), allow_pickle=True)
        if "test_pred_genomic" not in z.files:
            continue
        V.append(np.asarray(z["val_pred"], float).ravel())
        for s in sets:
            k = f"test_pred_{s}"
            T[s].append(np.asarray(z[k], float).ravel() if k in z.files else None)
    if not V:
        return None
    V = np.array(V).T
    keep = [s for s in sets if all(t is not None for t in T[s])]
    Tm = {s: np.array(T[s]).T for s in keep}
    stack, _, _ = enet_all_pos(V, yv, Tm, np.arange(len(yv)))

    out: dict[str, float] = {}
    for s in keep:
        out[s] = float(np.mean((stack[s] - Y[s]) ** 2))
    if "genomic" in stack:
        for s in PERTURB:
            if s in stack and len(stack[s]) == len(stack["genomic"]):
                dp = stack[s] - stack["genomic"]
                dy = Y[s] - Y["genomic"]
                out[f"D:{s}"] = float(np.mean((dp - dy) ** 2))
    if "snv_ref" in stack and "snv_alt" in stack:
        out["D:snv"] = float(
            np.mean(((stack["snv_alt"] - stack["snv_ref"]) - (Y["snv_alt"] - Y["snv_ref"])) ** 2)
        )
    out["_n_members"] = float(V.shape[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument(
        "--sets",
        default="genomic,ood,random_32k,sub_high,D:inversion,D:snv",
        help="endpoints to draw as curve panels",
    )
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    table: dict[str, dict[int, dict[str, float]]] = {}
    for res, suffix in RESERVOIRS:
        for D in sorted(TUNED):
            d = cell_dir(suffix, D)
            if d is None:
                continue
            m = endpoint_mse(d)
            if m:
                table.setdefault(res, {})[D] = m
                print(
                    f"{res:<20} D={D:>8,}  {TUNED[D]:<10} {int(m['_n_members'])} members",
                    flush=True,
                )
    if not table:
        print("no tuned cells found")
        return 1
    (out / "scaling_mse.json").write_text(json.dumps(table, indent=1, default=float))

    panels = [s.strip() for s in a.sets.split(",")]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0))
    for ax, ep in zip(axes.ravel(), panels, strict=False):
        for (res, _), colour in zip(RESERVOIRS, PALETTE, strict=False):
            pts = [
                (D, v[ep])
                for D, v in sorted(table.get(res, {}).items())
                if ep in v and np.isfinite(v[ep]) and v[ep] > 0
            ]
            if len(pts) < 2:
                continue
            xs = np.array([p[0] for p in pts], float)
            ys = np.array([p[1] for p in pts], float)
            ax.plot(xs, ys, "o-", color=colour, ms=5, lw=1.7, label=res)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(ep.replace("D:", "Δ"), fontsize=10)
        ax.grid(True, which="both", alpha=0.2, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("training sequences")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (stacked ensemble)")
    axes[0][0].legend(fontsize=7.5, ncols=2)
    fig.suptitle(
        "Scaling in MSE, tuned LegNet recipe at every size, across eval sets",
        y=0.98,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig12_scaling_mse.png", dpi=170)
    plt.close(fig)
    print("fig12 written")

    # ---- heatmaps ---------------------------------------------------------------
    eps = sorted(
        {e for r in table.values() for v in r.values() for e in v if not e.startswith("_")}
    )
    Ds = sorted({D for r in table.values() for D in r})
    res_present = [r for r, _ in RESERVOIRS if r in table]

    fig, axes = plt.subplots(
        1, 2, figsize=(14.5, max(4.0, 0.22 * len(eps))), gridspec_kw={"wspace": 0.45}
    )

    # (a) MSE at the largest common size, as a ratio to the best corpus on that endpoint.
    bigD = max(D for D in Ds if sum(D in table[r] for r in res_present) >= 3)
    M = np.full((len(eps), len(res_present)), np.nan)
    for j, r in enumerate(res_present):
        for i, e in enumerate(eps):
            M[i, j] = table[r].get(bigD, {}).get(e, np.nan)
    best = np.nanmin(M, axis=1, keepdims=True)
    R = M / best
    im = axes[0].imshow(R, aspect="auto", cmap="RdYlGn_r", vmin=1.0, vmax=np.nanpercentile(R, 95))
    axes[0].set_xticks(range(len(res_present)), res_present, rotation=40, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(eps)), [e.replace("D:", "Δ") for e in eps], fontsize=6.5)
    axes[0].set_title(f"(a) MSE relative to the best corpus, D={bigD:,}", fontsize=10)
    fig.colorbar(im, ax=axes[0], fraction=0.04, label="x best")

    # (b) log-log slope per endpoint per corpus: how fast each endpoint improves with data.
    S = np.full((len(eps), len(res_present)), np.nan)
    for j, r in enumerate(res_present):
        for i, e in enumerate(eps):
            pts = [(D, v[e]) for D, v in sorted(table[r].items()) if e in v and v[e] > 0]
            if len(pts) >= 3:
                S[i, j] = np.polyfit(
                    np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts]), 1
                )[0]
    lim = np.nanmax(np.abs(S)) if np.isfinite(S).any() else 1.0
    im = axes[1].imshow(S, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim)
    axes[1].set_xticks(range(len(res_present)), res_present, rotation=40, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(eps)), [e.replace("D:", "Δ") for e in eps], fontsize=6.5)
    axes[1].set_title("(b) log-log slope: more negative = improves faster", fontsize=10)
    fig.colorbar(im, ax=axes[1], fraction=0.04, label="d log MSE / d log N")

    fig.suptitle("Every corpus against every endpoint", y=1.0, fontsize=12)
    fig.savefig(out / "fig13_scaling_heatmaps.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("fig13 written")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
