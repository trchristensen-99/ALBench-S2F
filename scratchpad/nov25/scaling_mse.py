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
import os
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
# Extra cell roots, colon-separated. /grid/wsbs is at 100% inodes, so runs from Nov 25 on
# write their cells to /grid/koo instead; without this they are simply invisible to the
# figures and a size column reads as "not measured" when it was measured somewhere else.
# cell_dir() still takes the RICHEST matching cell, so a root listed here cannot quietly
# replace a wed41 cell with a thinner one of the same name.
ROOTS += [Path(p) for p in os.environ.get("ALBENCH_EXTRA_ROOTS", "").split(":") if p]
# 1M is deliberately absent: motif_shared_core is the only reservoir that reaches it, so a
# point there has nothing to be compared against and only stretches the axis.
TUNED = {
    10_000: "e800_wd4",
    30_000: "e500_wd2",
    100_000: "e500",
    300_000: "e300_wd05",
}
# Only reservoirs with a cell at EVERY size, so no curve has a hole in it. Audited
# 2026-11-25: these six all have >=10 members at 10k/30k/100k/300k. The ones dropped and
# why -- evoaug, gc_matched, dinuc_shuffle and motif_syntax_core exist only at 30k and 300k,
# so they contributed two-point curves that read as full ones; `evoaug_paper2023` is the
# EvoAug variant that does have every rung, so it stands in for that family.
# zoonomia_orthologs has full coverage but is excluded from the study (validation-chromosome
# breach, 15.4% exact duplicates).
RESERVOIRS = [
    ("genomic", ""),  # the bare cell name carries no suffix
    ("random", "_random"),
    ("motif_shared_core", "_motif_shared_core"),
    ("motif_ct_enriched", "_motif_ct_enriched"),
    ("evoaug_paper2023", "_evoaug_paper2023"),
    ("mutagenesis", "_mutagenesis"),
]
# "ood" is the stored key for the high-activity DESIGNED set.
PRETTY_EP = {
    "ood": "high-activity designed",
    "random_32k": "random",
    "D:snv_sig": "SNV effect (Gosai, high-effect)",
    "D:evoaug_mix": "EvoAug-style \u0394 (multi-op, real effects)",
    "D:inversion": "\u0394inversion",
    "motif_grammar": "motif grammar (varied syntax)",
    "motif_planted": "motif planted (one configuration)",
}
# Sidecar predictions for eval sets that did not exist when the cells were trained
# (scratchpad/nov25/rescore_extra_sets.py). Looked up by cell name under a root, or beside
# the cell -- the filesystem holding the cells ran out of inodes, so both have to work.
import os as _os  # noqa: E402

SIDECAR_ROOT = _os.environ.get("ALBENCH_SIDECAR_ROOT", "")
SIDECAR_NAME = "extra_evalsets_v1.npz"
# The operations pooled into the EvoAug-style endpoint. One perturbation on its own is a
# narrow question; EvoAug's augmentations are a MIX of substitution, indel and
# rearrangement, and the use case (a personal genome against the reference) is likewise.
EVOAUG_OPS = ("sub_med", "sub_high", "ins_med", "del_med", "inversion", "translocation")
EVOAUG_GATE = 1.0  # |true delta| in log2FC that an effect must clear to be scored
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


_SNV: dict | None = None


def snv_significant(z_gate: float = 2.0) -> dict | None:
    """Real MEASURED SNV effects whose size clears measurement error.

    The stored `D:snv` endpoint scores every pair, and 96% of pairs have an effect smaller
    than the noise on it (true delta sd 0.466, mean |delta| 0.285, only 4.0% above 1.0), so
    it is mostly a test of predicting zero -- which is why it discriminates at 1.6x. Gating
    on |delta| / SE(delta) makes it a test of predicting effects that are real.

    Two things make this stronger than the oracle-based delta endpoints: the target is the
    MEASUREMENT, not the oracle's opinion of it (the two agree at only r = 0.40 on deltas),
    and SE comes from Gosai's own lfcSE rather than being assumed. Row order is verified
    against the stored oracle labels before use.
    """
    global _SNV
    if _SNV is not None:
        return _SNV or None
    bat = REPO / "data/k562/test_sets_ag_s2_chrsplit/snv_mono_chr7_13_intermediate.npz"
    if not bat.exists():
        _SNV = {}
        return None
    b = np.load(bat, allow_pickle=True)
    ref_s = [str(x) for x in b["ref_sequences"]]
    alt_s = [str(x) for x in b["alt_sequences"]]
    td = np.asarray(b["true_alt_label"], float) - np.asarray(b["true_ref_label"], float)

    try:
        # Read the table directly rather than through load_gosai: that returns base INDICES
        # inside a plasmid window, and the battery stores bare 200-bp strings, so the join
        # key only exists in the raw file.
        # Same resolution order as albench.celltype.data.load_gosai; the table is not a
        # declared paths.py asset.
        import os

        import pandas as pd

        cand = [
            os.environ.get("ALBENCH_GOSAI_TABLE_S2", ""),
            str(REPO / "data/k562/DATA-Table_S2__MPRA_dataset.txt"),
        ]
        tbl_path = next((c for c in cand if c and os.path.exists(c)), "")
        if not tbl_path:
            raise FileNotFoundError("Gosai Table S2 not found")
        tbl = pd.read_csv(tbl_path, sep="\t", low_memory=False)
        se_by_seq = dict(
            zip(tbl["sequence"].astype(str), tbl["K562_lfcSE"].astype(float), strict=True)
        )
        sr = np.array([se_by_seq.get(x, np.nan) for x in ref_s])
        sa = np.array([se_by_seq.get(x, np.nan) for x in alt_s])
        sd = np.sqrt(sr**2 + sa**2)
        print(
            f"  SNV gate: SE found for {int(np.isfinite(sd).sum()):,} of {len(sd):,} pairs",
            flush=True,
        )
    except Exception as e:  # SE unavailable: fall back to a magnitude gate, and say so
        print(f"  SNV gate: no SE ({type(e).__name__}: {e}); using |delta| >= 1.0", flush=True)
        sd = np.full(len(td), 0.5)
    keep = np.isfinite(td) & np.isfinite(sd) & (np.abs(td) >= z_gate * sd)
    print(f"  SNV significant-effect gate: {int(keep.sum()):,} of {len(td):,} pairs", flush=True)
    _SNV = {"delta": td, "keep": keep}
    return _SNV


def cell_dir(res_suffix: str, D: int) -> Path | None:
    """The richest matching cell. Both roots can hold the same name with different seed
    counts (wed41 has 5 members at 30k where menu_retrain_v0 has 20), and taking whichever
    root was listed first would assemble one curve from two different ensembles."""
    name = f"bench38_ln_{TUNED[D]}_{D // 1000}k_c20{res_suffix}"
    best, best_n = None, 0
    for r in ROOTS:
        p = r / name
        n = len(glob.glob(str(p / "*_meta.json"))) if p.is_dir() else 0
        if n > best_n:
            best, best_n = p, n
    return best


def _extra_sets(d: Path, used: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Sidecar predictions for one cell, as {set: ((N, members), oracle)}."""
    cands = [Path(SIDECAR_ROOT) / f"{d.name}.npz"] if SIDECAR_ROOT else []
    cands.append(d / SIDECAR_NAME)
    p = next((c for c in cands if c.exists()), None)
    if p is None:
        return {}
    z = np.load(p, allow_pickle=True)
    mids = [str(x) for x in z["members"]]
    try:
        cols = [mids.index(u) for u in used]
    except ValueError:
        return {}  # scored before this cell gained the members the figure stacks
    out = {}
    for k in z.files:
        if not k.startswith("pred_"):
            continue
        s = k[len("pred_") :]
        out[s] = (np.asarray(z[k], float)[cols].T, np.asarray(z[f"oracle_{s}"], float).ravel())
    return out


def endpoint_mse(d: Path, max_members: int = 3) -> dict[str, float] | None:
    """Stacked-ensemble MSE on every level and effect endpoint for one trained cell.

    `max_members` is load-bearing, not tidiness. The cells hold 3 to 20 seeds depending on
    when they were run, and a bigger stack has lower MSE for reasons that have nothing to do
    with corpus size -- genomic has 20 members at 100k and 3 at 300k, which alone would bend
    the curve upward at the top. Every point is therefore built from the same number of
    members, the smallest any rung has.
    """
    lab_p = d / "labels.npz"
    metas = sorted(glob.glob(str(d / "*_meta.json")))
    if not metas or not lab_p.exists():
        return None
    lab = np.load(lab_p, allow_pickle=True)
    sets = [k[len("oracle_") :] for k in lab.files if k.startswith("oracle_")]
    Y = {s: np.asarray(lab[f"oracle_{s}"], float).ravel() for s in sets}
    yv = np.asarray(lab["val_labels"], float).ravel()

    V, T, used = [], {s: [] for s in sets}, []
    for m in metas[:max_members]:
        z = np.load(m.replace("_meta.json", ".npz"), allow_pickle=True)
        if "test_pred_genomic" not in z.files:
            continue
        V.append(np.asarray(z["val_pred"], float).ravel())
        used.append(Path(m).name[: -len("_meta.json")])
        for s in sets:
            k = f"test_pred_{s}"
            T[s].append(np.asarray(z[k], float).ravel() if k in z.files else None)
    if not V:
        return None
    V = np.array(V).T
    keep = [s for s in sets if all(t is not None for t in T[s])]
    Tm = {s: np.array(T[s]).T for s in keep}
    # Extra sets are stacked by the SAME validation-fitted weights as the trained-in ones,
    # so their columns have to be the same members in the same order; a sidecar that does
    # not cover every member used here is dropped rather than padded.
    for s, (p, y) in _extra_sets(d, used).items():
        Tm[s] = p
        Y[s] = y
    stack, _, _ = enet_all_pos(V, yv, Tm, np.arange(len(yv)))

    out: dict[str, float] = {}
    # Over `Tm`, not `keep`: the sidecar sets were added to Tm after keep was computed, and
    # scoring only `keep` silently dropped every rescored endpoint.
    for s in Tm:
        out[s] = float(np.mean((stack[s] - Y[s]) ** 2))
    if "genomic" in stack:
        for s in PERTURB:
            if s in stack and len(stack[s]) == len(stack["genomic"]):
                dp = stack[s] - stack["genomic"]
                dy = Y[s] - Y["genomic"]
                out[f"D:{s}"] = float(np.mean((dp - dy) ** 2))
    if "genomic" in stack:
        # Pooled EvoAug-style endpoint: several operations at once, scored only where the
        # true effect is large enough to be a real effect rather than a test of zero.
        dp, dy = [], []
        for s in EVOAUG_OPS:
            if s in stack and len(stack[s]) == len(stack["genomic"]):
                dp.append(stack[s] - stack["genomic"])
                dy.append(Y[s] - Y["genomic"])
        if dp:
            dpc, dyc = np.concatenate(dp), np.concatenate(dy)
            k = np.abs(dyc) >= EVOAUG_GATE
            if k.sum() > 200:
                out["D:evoaug_mix"] = float(np.mean((dpc[k] - dyc[k]) ** 2))
    if "snv_ref" in stack and "snv_alt" in stack:
        pd_ = stack["snv_alt"] - stack["snv_ref"]
        out["D:snv"] = float(np.mean((pd_ - (Y["snv_alt"] - Y["snv_ref"])) ** 2))
        g = snv_significant()
        if g is not None and len(g["delta"]) == len(pd_):
            k = g["keep"]
            if k.sum() > 200:
                out["D:snv_sig"] = float(np.mean((pd_[k] - g["delta"][k]) ** 2))
    out["_n_members"] = float(V.shape[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/nov25")
    ap.add_argument("--max-members", type=int, default=3)
    ap.add_argument(
        "--sets",
        # Not motif_grammar: it is scored from sidecars, which only the checkpointed wed41
        # cells have, and those exist at two sizes -- it would be a two-point panel.
        default="genomic,ood,random_32k,sub_high,D:evoaug_mix,D:snv_sig",
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
            m = endpoint_mse(d, a.max_members)
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
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0))
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
        ax.set_title(PRETTY_EP.get(ep, ep.replace("D:", "Δ")), fontsize=10)
        ax.grid(True, which="both", alpha=0.2, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("training sequences")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (stacked ensemble)")
    # Below the panels: inside the first one it sat on the curves it was labelling.
    h, lb = axes[0][0].get_legend_handles_labels()
    fig.legend(
        h, lb, loc="lower center", ncols=min(6, len(lb)), fontsize=8.5, bbox_to_anchor=(0.5, 0.0)
    )
    fig.suptitle(
        "Scaling in MSE, tuned LegNet recipe at every size, across eval sets",
        y=0.98,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
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
