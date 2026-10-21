"""Evidence generator for the pipeline-design artifact. Reads experiment outputs, writes figures (PNG) and data.json.

Re-run any time: every figure recomputes from whatever has finished, labels the number of models it used, and marks arms that are
still running. Nothing here is hand-typed except layout.
    python make_justification.py --out outputs/justification
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import ElasticNetCV  # noqa: E402

R = Path("/grid/wsbs/home_norepl/christen/ALBench-S2F")
os.chdir(R)
EV = ("genomic", "ood", "random_32k")
INK, MUTED, GRID = "#14202e", "#5a6878", "#e3e8ee"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE, TEAL = "#1d6fa5", "#d9822b", "#2a7d4f", "#b5483a", "#8b97a6", "#6a4fa3", "#12857c"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#b9c3cf", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 9.5, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 8.5})
DATA: dict = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}


# ------------------------------------------------------------------------------------------------ loading and ensembles
def pear(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def load_cell(d, pattern="r*_meta.json"):
    """One cell dir -> dict(V,T,yv,yt,meta) for models with finite predictions; None when no labels/models."""
    d = str(d)
    lp = d + "/labels.npz"
    if not os.path.exists(lp):
        return None
    lab = np.load(lp)
    if "val_labels" not in lab.files:
        return None
    yv = lab["val_labels"].astype(float)
    yt = {e: lab[f"oracle_{e}"].astype(float) for e in EV if f"oracle_{e}" in lab.files}
    rows = []
    for f in glob.glob(d + "/" + pattern):
        if f.endswith("_screen.json"):
            continue
        try:
            m = json.load(open(f))
        except Exception:
            continue
        if "error" in m:
            continue
        npz = f.replace("_meta.json", ".npz")
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        if "val_pred" not in z.files or not all(f"test_pred_{e}" in z.files for e in yt):
            continue
        vp = z["val_pred"].astype(float)
        tp = {e: z[f"test_pred_{e}"].astype(float) for e in yt}
        if not (np.isfinite(vp).all() and all(np.isfinite(t).all() for t in tp.values())) or np.abs(vp).max() > 1e3:
            continue  # diverged / exploded models never enter an ensemble
        rows.append((int(m.get("round", 0)), m.get("model_id", f), vp, tp, m))
    if not rows:
        return None
    rows.sort(key=lambda r: (r[0], str(r[1])))
    V = np.column_stack([r[2] for r in rows])
    T = {e: np.column_stack([r[3][e] for r in rows]) for e in yt}
    return dict(V=V, T=T, yv=yv, yt=yt, rounds=np.array([r[0] for r in rows]), meta=[r[4] for r in rows],
                val=np.array([pear(V[:, j], yv) for j in range(V.shape[1])]))


_SUB = {}


def subidx(n):
    if n not in _SUB:
        _SUB[n] = np.random.default_rng(0).choice(n, min(n, 8000), replace=False)
    return _SUB[n]


def enet(V, yv, T, yt, cols):
    """Scale-robust enet_all_pos (z-scored columns, positive weights) -> {set: test r}, n kept."""
    cols = list(cols)
    sub = subidx(len(yv))
    mu, sd = V[sub][:, cols].mean(0), V[sub][:, cols].std(0) + 1e-12
    en = ElasticNetCV(cv=5, n_alphas=40, max_iter=5000, l1_ratio=[.1, .5, .9], positive=True).fit((V[sub][:, cols] - mu) / sd, yv[sub])
    return {e: pear(en.predict((T[e][:, cols] - mu) / sd), yt[e]) for e in yt}, int((en.coef_ > 0).sum())


def pooled(cells, cols_fn=None):
    """Concatenate several cells that share labels (same val/test sets)."""
    cells = [c for c in cells if c is not None]
    if not cells:
        return None
    V = np.column_stack([c["V"] for c in cells])
    T = {e: np.column_stack([c["T"][e] for c in cells]) for e in cells[0]["T"]}
    return dict(V=V, T=T, yv=cells[0]["yv"], yt=cells[0]["yt"])


def save(fig, name, out):
    p = Path(out) / "figs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", p.name, flush=True)


def jitter(n, w=0.08, seed=0):
    return np.random.default_rng(int(seed * 10)).uniform(-w, w, n)


def note(ax, text, loc=(0.02, 0.98), **kw):
    ax.text(*loc, text, transform=ax.transAxes, va="top", ha="left", fontsize=8.3, color=MUTED, **kw)


# ------------------------------------------------------------------------------------------------ 1. training / evaluation loop
def fig_loop(out):
    p = R / "outputs/harness_ab_oct20/main2/results.jsonl"
    rows = [json.loads(l) for l in open(p)] if p.exists() else []
    rows = [r for r in rows if "error" not in r]
    by = collections.defaultdict(list)
    for r in rows:
        by[(str(r.get("cfg", "F%s" % r.get("arch"))), r["arm"])].append(r)

    def mean_se(c, a, k):
        v = [x[k] for x in by.get((c, a), [])]
        return (np.mean(v), (np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0), v) if v else (np.nan, 0, [])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    series = [("native loop, forward-only\n(what the pipeline reported)", "S_native", "gen_fwd", GREY),
              ("native loop + RC averaging", "S_native", "gen_r", "#a9c6dc"),
              ("arch loop, forward-only", "A_arch", "gen_fwd", "#7fb3d5"), ("arch loop + RC averaging", "A_arch", "gen_r", BLUE)]
    w = 0.2
    for si, (lab, arm, k, col) in enumerate(series):
        xs, ms = [], []
        for ci, c in enumerate(("0", "1", "2")):
            m, se, v = mean_se(c, arm, k)
            ax.bar(ci + (si - 1.5) * w, m, w * 0.92, color=col, label=lab if ci == 0 else None, zorder=2)
            ax.scatter(np.full(len(v), ci + (si - 1.5) * w) + jitter(len(v), 0.03, ci + si), v, s=9, color=INK, alpha=0.55, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"LegNet config {c}" for c in "012"])
    ax.set_ylim(0.66, 0.775); ax.set_ylabel("test Pearson r, genomic 30k (single model)")
    ax.legend(loc="upper left", ncols=2, fontsize=7.8)
    ax.set_title("Same network, same HPs, two training loops")
    gaps = {k: np.mean([mean_se(c, "A_arch", "gen_r")[0] - mean_se(c, "S_native", "gen_fwd")[0] for c in "012"]) for k in ("total",)}
    rc = np.mean([mean_se(c, "S_native", "gen_r")[0] - mean_se(c, "S_native", "gen_fwd")[0] for c in "012"])
    loop = np.mean([mean_se(c, "A_arch", "gen_fwd")[0] - mean_se(c, "S_native", "gen_fwd")[0] for c in "012"])
    DATA["loop"] = dict(total_gap=float(gaps["total"]), rc_avg=float(rc), loop_fwd_matched=float(loop),
                        native_reported=[float(mean_se(c, "S_native", "gen_fwd")[0]) for c in "012"],
                        arch_reported=[float(mean_se(c, "A_arch", "gen_r")[0]) for c in "012"], n_records=len(rows))
    ax = axes[1]
    base_m, base_se, _ = mean_se("1", "A_arch", "gen_r")
    items = [("A_flip_shift", "native per-sample crop shift"), ("A_flip_amp", "fp16 mixed precision"), ("A_flip_es", "Pearson early-stop + plateau 5"),
             ("A_flip_clip", "gradient clip = HP value (0)"), ("A_flip_cap", "no 25-min cap"), ("A_scal", "all of the above together")]
    ds = []
    for arm, lab in items:
        m, se, v = mean_se("1", arm, "gen_r")
        ds.append((lab, m - base_m, float(np.hypot(se, base_se)), len(v)))
    y = np.arange(len(ds))[::-1]
    ax.barh(y, [d[1] for d in ds], xerr=[d[2] for d in ds], color=[RED if d[1] < -0.003 else GREY for d in ds], height=0.6, zorder=2, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([f"{d[0]} (n={d[3]})" for d in ds], fontsize=8.5)
    ax.axvline(0, color=INK, lw=0.8)
    sn = mean_se("1", "S_native", "gen_r")[0] - base_m
    ax.axvline(sn, color=BLUE, ls="--", lw=1.1); ax.text(sn - 0.0003, -0.62, "old loop (RC-averaged)", color=BLUE, fontsize=8, ha="right", va="bottom")
    ax.set_ylim(-0.75, len(ds) - 0.4)
    ax.set_xlabel("change in test r vs the arch loop (config 1, RC-averaged)")
    ax.set_title("Which loop ingredient costs what")
    DATA["loop"]["flips"] = {a: float(d[1]) for (a, _), d in zip(items, ds)}
    fig.tight_layout()
    save(fig, "loop_decomposition.png", out)

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    groups = [("LegNet-eff configs\n(HP search winners)", [("A_arch", c, BLUE) for c in "012"]), ("free-form architectures\n(top-3 by val)", [("F_arch", f"F{i}", ORANGE) for i in range(3)])]
    xi, ticks, labs = 0, [], []
    allm = {}
    for gname, members in groups:
        start = xi
        for arm, c, col in members:
            m, se, v = mean_se(c, arm, "gen_r")
            ax.errorbar(xi, m, yerr=se, fmt="D", color=col, ms=6, capsize=3, zorder=3)
            ax.scatter(np.full(len(v), xi) + jitter(len(v), 0.06, xi), v, s=14, color=col, alpha=0.5, zorder=2)
            allm.setdefault(gname, []).append(m); xi += 1
        ticks.append((start + xi - 1) / 2); labs.append(gname); xi += 0.8
    ax.set_xticks(ticks); ax.set_xticklabels(labs)
    ax.set_ylabel("test Pearson r (RC-averaged), genomic 30k"); ax.set_ylim(0.715, 0.745)
    ax.set_title("Inside one training loop, free-form architectures do not beat LegNet")
    DATA["arch_vs_legnet"] = {k.split("\n")[0]: [float(x) for x in v] for k, v in allm.items()}
    fig.tight_layout()
    save(fig, "arch_vs_legnet.png", out)


# ------------------------------------------------------------------------------------------------ 2. menu retraining across D
def load_menu(tag_dirs):
    ent = {}
    yv = yt = None
    for d in tag_dirs:
        if not os.path.exists(f"{d}/labels.npz"):
            continue
        lab = np.load(f"{d}/labels.npz")
        yv = lab["val_labels"].astype(float)
        yt = {e: lab[f"oracle_{e}"].astype(float) for e in EV}
        for f in glob.glob(f"{d}/r*_menu_*_meta.json"):
            m = json.load(open(f))
            z = np.load(f.replace("_meta.json", ".npz"))
            vp = z["val_pred"].astype(float)
            if not np.isfinite(vp).all() or np.abs(vp).max() > 1e3:
                continue
            ent.setdefault(m["menu_entry"], []).append(dict(
                val=pear(vp, yv), vp=vp, tp={e: z[f"test_pred_{e}"].astype(float) for e in EV},
                test=m["test_oracle_pearson"], idea=m.get("idea", "")))
    return yv, yt, ent


def fig_menu(out):
    base = R / "outputs/menu_retrain_v0"
    tags = {"10k": ["g10k", "g10k_seeds12"], "30k": ["g30k", "g30k_seeds12"], "100k": ["g100k", "g100k_seeds12"]}
    data = {k: load_menu([str(base / t) for t in v]) for k, v in tags.items()}
    if not all(d[2] for d in data.values()):
        return
    stat = {k: {e: dict(val=np.mean([r["val"] for r in v]), test=np.mean([r["test"] for r in v]), n=len(v)) for e, v in d[2].items()} for k, d in data.items()}
    common = sorted(set.intersection(*[set(s) for s in stat.values()]))
    nseeds = {k: max(s["n"] for s in stat[k].values()) for k in stat}
    DATA["menu"] = dict(n_entries=len(common), seeds_per_D=nseeds)
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2))
    for ax, (a, b) in zip(axes[0], (("10k", "30k"), ("30k", "100k"))):
        va = np.array([stat[a][c]["val"] for c in common]); vb = np.array([stat[b][c]["val"] for c in common])
        ta = np.array([stat[a][c]["test"] for c in common]); tb = np.array([stat[b][c]["test"] for c in common])
        isref = np.array([c == "reference_legnet_eff" for c in common])
        ax.scatter(va[~isref], vb[~isref], s=26, color=BLUE, alpha=0.75, zorder=3)
        ax.scatter(va[isref], vb[isref], s=80, color=ORANGE, marker="*", zorder=4, label="reference LegNet-eff")
        rs, rt = spearmanr(va, vb)[0], spearmanr(ta, tb)[0]
        ax.set_xlabel(f"validation r at D={a}"); ax.set_ylabel(f"validation r at D={b}")
        ax.set_title(f"Same {len(common)} architectures, D={a} vs D={b}")
        note(ax, f"Spearman: validation {rs:.2f}, test {rt:.2f}")
        ax.legend(loc="lower right")
        DATA["menu"][f"spearman_{a}_{b}"] = dict(val=float(rs), test=float(rt))
    a10, a100 = "10k", "100k"
    DATA["menu"]["spearman_10k_100k"] = dict(val=float(spearmanr([stat[a10][c]["val"] for c in common], [stat[a100][c]["val"] for c in common])[0]),
                                             test=float(spearmanr([stat[a10][c]["test"] for c in common], [stat[a100][c]["test"] for c in common])[0]))
    ax = axes[1][0]
    ranks, beat = [], []
    for k in ("10k", "30k", "100k"):
        order = sorted(common, key=lambda c: -stat[k][c]["test"])
        ranks.append(order.index("reference_legnet_eff") + 1); beat.append(ranks[-1] - 1)
    ax.plot(range(3), ranks, "-o", color=ORANGE, lw=2)
    for i, (rk, bt) in enumerate(zip(ranks, beat)):
        ax.annotate(f"rank {rk}/{len(common)}\n{bt} free-form models beat it", (i, rk), textcoords="offset points", xytext=(9, 0) if i == 0 else (0, -24), ha="left" if i == 0 else "center", va="center", fontsize=8.3)
    ax.set_xticks(range(3)); ax.set_xticklabels(["D = 10k", "D = 30k", "D = 100k"]); ax.set_ylim(len(common) + 2, -2); ax.set_xlim(-0.3, 2.9)
    ax.set_ylabel("test-r rank of reference LegNet (1 = best)")
    ax.set_title("A fixed LegNet moves up the ranking as D grows")
    DATA["menu"]["reference_rank"] = dict(zip(("10k", "30k", "100k"), map(int, ranks)))
    ax = axes[1][1]
    ens_all, ens_top5, ens_hp = [], [], []
    top5 = sorted(common, key=lambda c: -stat["10k"][c]["val"])[:5]
    for k in ("10k", "30k", "100k"):
        yv, yt, ent = data[k]
        for keys, store in ((common, ens_all), (top5, ens_top5)):
            V = np.column_stack([np.mean([r["vp"] for r in ent[c]], 0) for c in keys])
            T = {e: np.column_stack([np.mean([r["tp"][e] for r in ent[c]], 0) for c in keys]) for e in EV}
            store.append(enet(V, yv, T, yt, range(V.shape[1]))[0]["genomic"])
    for k, D in (("10k", 10000), ("30k", 30000), ("100k", 100000)):
        cells = [load_cell(R / f"outputs/scaling_rep_oct16/genomic_d{D}/rep{r}") for r in (1, 2, 3)]
        pc = pooled(cells)
        ens_hp.append(enet(pc["V"], pc["yv"], pc["T"], pc["yt"], range(pc["V"].shape[1]))[0]["genomic"] if pc else np.nan)
    x = np.arange(3)
    ax.plot(x, ens_all, "-o", color=BLUE, lw=2, label=f"retrained menu, all {len(common)} (arch loop)")
    ax.plot(x, ens_top5, "-s", color=TEAL, lw=1.6, label="top-5 picked at D=10k, retrained")
    ax.plot(x, ens_hp, "--^", color=GREY, lw=1.6, label="HP-only search, 3 runs pooled (old loop)")
    ax.set_xticks(x); ax.set_xticklabels(["D = 10k", "D = 30k", "D = 100k"]); ax.set_ylabel("ensemble test r, genomic")
    ax.legend(loc="upper left", fontsize=8); ax.set_title("Retrain the whole pool at every D")
    DATA["menu"]["ensembles"] = dict(all=[float(v) for v in ens_all], top5_at_10k=[float(v) for v in ens_top5], hp_only=[float(v) for v in ens_hp])
    fig.tight_layout()
    save(fig, "menu_across_D.png", out)


# ------------------------------------------------------------------------------------------------ 3. instance length, top-k, instances
def fig_instances(out):
    inst = {r: load_cell(R / f"outputs/archsearch_final_oct20/son55/rep{r}") for r in (1, 2, 3)}
    inst = {k: v for k, v in inst.items() if v is not None}
    if len(inst) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    ax = axes[0]
    ns = [5, 10, 15, 20, 30, 40, 50]
    curves = {}
    for rep, c in inst.items():
        xs, ys = [], []
        for n in ns + [c["V"].shape[1]]:
            if n <= c["V"].shape[1]:
                xs.append(n); ys.append(enet(c["V"], c["yv"], c["T"], c["yt"], range(n))[0]["genomic"])
        curves[rep] = (xs, ys)
        ax.plot(xs, ys, "-o", ms=3.5, lw=1.4, alpha=0.8, label=f"independent run {rep}")
    common = [n for n in ns if all(n in c[0] for c in curves.values())]
    mean = [np.mean([dict(zip(*curves[r]))[n] for r in curves]) for n in common]
    ax.plot(common, mean, "-", color=INK, lw=2.4, label="mean of runs")
    ax.set_xlabel("models trained in one AutoResearch run (2 per round)"); ax.set_ylabel("ensemble test r, genomic 30k")
    ax.set_title("One run saturates quickly"); ax.legend(fontsize=8)
    DATA["instances"] = dict(n=common, mean=[float(m) for m in mean], sd_between_runs=float(np.std([ys[-1] for _, ys in curves.values()], ddof=1)))
    ax = axes[1]
    ks = [1, 2, 3, 5, 8, 12, "all"]
    res = []
    for k in ks:
        Vs, Ts = [], {e: [] for e in EV}
        for rep, c in inst.items():
            idx = np.argsort(-c["val"])[: (c["V"].shape[1] if k == "all" else k)]
            Vs.append(c["V"][:, idx]); [Ts[e].append(c["T"][e][:, idx]) for e in EV]
        V = np.column_stack(Vs); T = {e: np.column_stack(Ts[e]) for e in EV}
        c0 = next(iter(inst.values()))
        res.append(enet(V, c0["yv"], T, c0["yt"], range(V.shape[1]))[0]["genomic"])
    ax.bar(range(len(ks)), res, color=BLUE, zorder=2)
    for i, v in enumerate(res):
        ax.text(i, v + 0.0008, f"{v:.3f}", ha="center", fontsize=8)
    singles = [enet(c["V"], c["yv"], c["T"], c["yt"], range(c["V"].shape[1]))[0]["genomic"] for c in inst.values()]
    ax.axhspan(min(singles), max(singles), color=ORANGE, alpha=0.18, zorder=1)
    ax.text(-0.4, max(res + singles) + 0.0045, "shaded: range of single runs using all of their own models", color=ORANGE, fontsize=8, ha="left")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([str(k) for k in ks]); ax.set_ylim(min(res + singles) - 0.012, max(res + singles) + 0.006)
    ax.set_xlabel("top-k models by validation kept from each of 3 runs"); ax.set_ylabel("pooled ensemble test r")
    ax.set_title("Top 8 per run already equals the whole pool")
    DATA["topk"] = dict(ks=[str(k) for k in ks], ens=[float(v) for v in res], singles=[float(s) for s in singles])
    fig.tight_layout()
    save(fig, "instances_and_topk.png", out)


# ------------------------------------------------------------------------------------------------ 4. prompt arms / proposer comparison
ARMS = {"P  guidance + build-on-what-works": ("son55", BLUE), "W  guidance only": ("son55_wide", TEAL), "N  nothing": ("son55_none", ORANGE)}
PROP = {"Sonnet 5.5": ("son55", BLUE), "Sonnet 4.5": ("son45", PURPLE), "Opus 5.5": ("opus55", RED)}


def arm_figure(groups, title, fname, key, out, family_dir="archsearch_final_oct20"):
    cells = {}
    for gname, (sub, col) in groups.items():
        for rep in (1, 2, 3):
            c = load_cell(R / f"outputs/{family_dir}/{sub}/rep{rep}")
            if c is not None and c["V"].shape[1] >= 5:
                cells[(gname, rep)] = c
    if not cells:
        return
    n_eq = max(10, int(min(c["V"].shape[1] for c in cells.values()) // 10 * 10))
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))
    ax = axes[0]
    summary = {}
    for gi, (gname, (sub, col)) in enumerate(groups.items()):
        pts = [(rep, c) for (g, rep), c in cells.items() if g == gname]
        if not pts:
            continue
        t5 = [np.sort(c["val"])[-5:].mean() for _, c in pts]
        ax.scatter(np.full(len(t5), gi) + jitter(len(t5), 0.07, gi), t5, s=34, color=col, zorder=3)
        for (rep, c), v in zip(pts, t5):
            ax.annotate(f"n={c['V'].shape[1]}", (gi + 0.1, v), fontsize=7, color=MUTED)
        summary.setdefault(gname, {})["top5_val"] = [float(v) for v in t5]
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g.split("  ")[0] if "  " in g else g for g in groups], fontsize=9)
    ax.set_ylabel("mean validation r of the 5 best models"); ax.set_title("Best models found (as run so far)")
    ax = axes[1]
    for gi, (gname, (sub, col)) in enumerate(groups.items()):
        for (g, rep), c in cells.items():
            if g != gname:
                continue
            early = c["val"][c["rounds"] <= 5]; late = c["val"][c["rounds"] >= 6]
            if len(early) and len(late):
                ax.plot([gi - 0.12, gi + 0.12], [early.mean(), late.mean()], "-o", color=col, ms=4, alpha=0.8)
                summary.setdefault(gname, {}).setdefault("early_late", []).append([float(early.mean()), float(late.mean())])
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g.split("  ")[0] if "  " in g else g for g in groups], fontsize=9)
    ax.set_ylabel("mean validation r of proposals"); ax.set_title("Rounds 1-6 (left) vs later (right)")
    ax = axes[2]
    for gi, (gname, (sub, col)) in enumerate(groups.items()):
        e = []
        for (g, rep), c in cells.items():
            if g == gname and c["V"].shape[1] >= n_eq:
                e.append(enet(c["V"], c["yv"], c["T"], c["yt"], range(n_eq))[0]["genomic"])
        if e:
            ax.scatter(np.full(len(e), gi) + jitter(len(e), 0.07, gi + 5), e, s=34, color=col, zorder=3)
            ax.hlines(np.mean(e), gi - 0.22, gi + 0.22, color=col, lw=2)
            summary.setdefault(gname, {})["ens_at_n"] = [float(v) for v in e]
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g.split("  ")[0] if "  " in g else g for g in groups], fontsize=9)
    ax.set_ylabel(f"ensemble test r at equal n = {n_eq} models"); ax.set_title("Equal-size ensembles")
    fig.suptitle(title, y=1.02, fontsize=11, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, fname, out)
    DATA[key] = dict(n_eq=n_eq, summary=summary, cells={f"{g}|{r}": int(c["V"].shape[1]) for (g, r), c in cells.items()})


# ------------------------------------------------------------------------------------------------ 5. LLM vs classical, KB effect, replicates, context pilot
def fig_llm_vs_classical(out):
    res = {}
    arms = [("llm2x30", BLUE, "LLM 2x30"), ("llm6x10", TEAL, "LLM 6x10"), ("llm1x60", PURPLE, "LLM 1x60"), ("multi6", GREY, "6 classical samplers")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    ax = axes[0]
    for ri, rname in enumerate(("genomic", "motif_ct_enriched", "random")):
        for ai, (arm, col, lab) in enumerate(arms):
            c = load_cell(R / f"outputs/cadence/{rname}_{arm}")
            if c is None:
                continue
            k = min(60, c["V"].shape[1])
            v = enet(c["V"], c["yv"], c["T"], c["yt"], range(k))[0]["genomic"]
            res[(rname, arm)] = (v, k)
            ax.bar(ri + (ai - 1.5) * 0.2, v, 0.19, color=col, label=lab if ri == 0 else None, zorder=2)
            ax.text(ri + (ai - 1.5) * 0.2, v + 0.002, str(k), ha="center", fontsize=6.8, color=MUTED)
    ax.set_xticks(range(3)); ax.set_xticklabels(["genomic", "motif", "random"]); ax.set_ylim(0.66, 0.84)
    ax.set_ylabel("ensemble test r (<=60 models, 30k)"); ax.set_title("Oct-13 cadence study (1 run per cell)"); ax.legend(fontsize=7.6, loc="upper left", ncols=2)
    DATA["cadence"] = {f"{a}|{b}": [float(v), int(n)] for (a, b), (v, n) in res.items()}
    ax = axes[1]
    for arm, col, lab in (("llm2x30", BLUE, "LLM 2x30"), ("multi6", GREY, "classical")):
        c = load_cell(R / f"outputs/cadence/genomic_{arm}")
        if c is None:
            continue
        r = c["rounds"]; mx = r.max()
        xs = sorted(set(r)); ys = [c["val"][r == x].mean() for x in xs]
        ax.plot(xs, ys, "-o", ms=3, color=col, label=lab)
    ax.set_xlabel("round"); ax.set_ylabel("mean validation r of proposals"); ax.set_title("LLM proposals are good from round 0"); ax.legend()
    ax = axes[2]
    base = R / "outputs/baseline_oct20"
    done = {}
    for arm in ("random_broad", "prior_marginal", "prior_tpe"):
        done[arm] = [len([f for f in glob.glob(str(base / arm / f"run{k}/r*_meta.json")) if not f.endswith("_screen.json")]) for k in range(1, 7)]
    labs = [("random_broad", "random,\nfull box", GREY), ("prior_marginal", "random,\nLLM-like prior", TEAL), ("prior_tpe", "TPE,\nLLM-like box", ORANGE)]
    for i, (arm, lab, col) in enumerate(labs):
        ax.barh(i, sum(done[arm]), color=col, height=0.6, zorder=2)
        ax.text(sum(done[arm]) + 3, i, f"{sum(done[arm])} / 180", va="center", fontsize=8.5)
    ax.set_yticks(range(3)); ax.set_yticklabels([l[1] for l in labs], fontsize=8.5); ax.invert_yaxis(); ax.set_xlim(0, 200)
    ax.set_xlabel("models trained so far (6 runs x 30 per arm)"); ax.set_title("Repaired baselines: progress")
    DATA["baselines_progress"] = done
    fig.tight_layout()
    save(fig, "llm_vs_classical.png", out)


def fig_kb_replicates(out):
    fig, axes2 = plt.subplots(1, 2, figsize=(9.2, 4.1))
    figr, axr = plt.subplots(figsize=(6.6, 4.1))
    axes = [axes2[0], axes2[1], axr]
    ax = axes[0]
    cells = ["genomic_d10000", "genomic_d30000", "genomic_d100000", "motif_ct_enriched_d30000", "random_d30000"]
    labs = ["genomic\n10k", "genomic\n30k", "genomic\n100k", "motif\n30k", "random\n30k"]
    diffs = []
    for c in cells:
        a, b = load_cell(R / f"outputs/grid_oct15/default_{c}"), load_cell(R / f"outputs/grid_oct15/nokb_{c}")
        if a is None or b is None:
            diffs.append(np.nan); continue
        diffs.append(enet(a["V"], a["yv"], a["T"], a["yt"], range(a["V"].shape[1]))[0]["genomic"] - enet(b["V"], b["yv"], b["T"], b["yt"], range(b["V"].shape[1]))[0]["genomic"])
    ax.bar(range(5), diffs, color=[GREEN if d > 0 else RED for d in diffs], zorder=2)
    ax.axhspan(-0.009, 0.009, color=GREY, alpha=0.2, zorder=1); ax.text(-0.45, -0.0087, "grey band: single-run noise (±0.009)", ha="left", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8); ax.set_ylabel("ensemble r: with KB minus without KB")
    ax.set_title("Knowledge-base prior (HP-only, 1 run/cell)")
    DATA["kb"] = dict(diffs=[None if np.isnan(d) else float(d) for d in diffs], mean=float(np.nanmean(diffs)))
    ax = axes[1]
    for vi, (v, col) in enumerate((("default", BLUE), ("nokb", GREY), ("informed", TEAL), ("critic", RED), ("explore", ORANGE), ("reflect", PURPLE))):
        pts = []
        for rep in (1, 2):
            c = load_cell(R / f"outputs/promptvar_oct16/{v}_d30000/rep{rep}")
            if c is not None:
                pts.append(enet(c["V"], c["yv"], c["T"], c["yt"], range(c["V"].shape[1]))[0]["genomic"])
        if pts:
            ax.scatter(np.full(len(pts), vi) + jitter(len(pts), 0.05, vi), pts, s=30, color=col, zorder=3)
            ax.hlines(np.mean(pts), vi - 0.2, vi + 0.2, color=col, lw=2)
    ax.set_xticks(range(6)); ax.set_xticklabels(["default", "no KB", "informed", "critic", "explore", "reflect"], rotation=30, fontsize=8)
    ax.set_ylabel("ensemble test r, genomic 30k"); ax.set_title("Prompt wording (HP-only, 2 reps each)")
    ax = axes[2]
    cols = {"genomic": BLUE, "motif_ct_enriched": TEAL, "random": ORANGE}
    xs, rows = 0, []
    for D in (10000, 30000, 100000):
        for rname, col in cols.items():
            cells_ = [load_cell(R / f"outputs/scaling_rep_oct16/{rname}_d{D}/rep{r}") for r in (1, 2, 3)]
            singles = [enet(c["V"], c["yv"], c["T"], c["yt"], range(c["V"].shape[1]))[0]["genomic"] for c in cells_ if c is not None]
            pc = pooled(cells_)
            if not singles or pc is None:
                continue
            pv = enet(pc["V"], pc["yv"], pc["T"], pc["yt"], range(pc["V"].shape[1]))[0]["genomic"]
            ax.scatter(np.full(len(singles), xs) + jitter(len(singles), 0.07, xs), singles, s=14, color=col, alpha=0.55, zorder=2)
            ax.scatter([xs], [pv], marker="D", s=44, color=col, edgecolor=INK, zorder=3)
            rows.append(dict(res=rname, D=D, singles=[float(s) for s in singles], pooled=float(pv))); xs += 1
        xs += 1
    ax.set_xticks([1, 5, 9]); ax.set_xticklabels(["D = 10k", "D = 30k", "D = 100k"]); ax.set_ylabel("ensemble test r, genomic eval")
    ax.set_title("3 independent runs: pooled (diamond) vs single")
    h = [plt.Line2D([], [], marker="o", ls="", color=c, label=n) for n, c in cols.items()]
    ax.legend(handles=h, fontsize=8, loc="upper left")
    DATA["replicates"] = rows
    fig.tight_layout(); figr.tight_layout()
    save(fig, "kb_prompts.png", out); save(figr, "replicates.png", out)


def fig_context(out):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), gridspec_kw=dict(width_ratios=[1, 1.1, 1.4]))
    ax = axes[0]
    names = ["C0 none", "C1 full\nhistory", "C2 warm\nstart", "C3 reservoir\ncontext"]
    vals = []
    for i, c in enumerate(("c0", "c1", "c2", "c3")):
        cc = load_cell(R / f"outputs/curriculum_oct14/{c}_30k")
        vals.append(enet(cc["V"], cc["yv"], cc["T"], cc["yt"], range(cc["V"].shape[1]))[0]["genomic"] if cc else np.nan)
    ax.bar(range(4), vals, color=[GREY, BLUE, TEAL, ORANGE], zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0006, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0.72, 0.745); ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("ensemble test r, genomic 30k")
    ax.set_title("HP-only pilot at 30k (1 run each)")
    DATA["ctx_pilot_30k"] = [None if np.isnan(v) else float(v) for v in vals]
    ax = axes[1]
    items = [("warm\nc2", "outputs/chain_oct15/c2_100k", BLUE), ("warm\nc5 (rerun)", "outputs/chain_oct15/c5_100k", BLUE), ("warm\nc3", "outputs/chain_oct15/c3_100k", TEAL),
             ("no ctx\nrep1", "outputs/scaling_rep_oct16/genomic_d100000/rep1", GREY), ("no ctx\nrep2", "outputs/scaling_rep_oct16/genomic_d100000/rep2", GREY),
             ("no ctx\nrep3", "outputs/scaling_rep_oct16/genomic_d100000/rep3", GREY), ("no ctx\ngrid", "outputs/grid_oct15/default_genomic_d100000", GREY)]
    vs = []
    for lab, d, col in items:
        c = load_cell(R / d)
        v = enet(c["V"], c["yv"], c["T"], c["yt"], range(min(16, c["V"].shape[1])))[0]["genomic"] if c else np.nan
        vs.append(v)
    ax.bar(range(len(items)), vs, color=[i[2] for i in items], zorder=2)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([i[0] for i in items], fontsize=6.8)
    ax.set_ylim(0.74, 0.81); ax.set_ylabel("ensemble test r (first 16 models), genomic 100k"); ax.set_title("Warm start at 100k did not replicate")
    DATA["ctx_pilot_100k"] = {i[0].replace("\n", " "): (None if np.isnan(v) else float(v)) for i, v in zip(items, vs)}
    ax = axes[2]
    arms = collections.OrderedDict([("none", GREY), ("T", BLUE), ("J", TEAL), ("X", ORANGE)])
    cells = {(a, r): load_cell(R / f"outputs/ctx100_oct20/{a}/rep{r}") for a in arms for r in (1, 2, 3, 4)}
    cells = {k: v for k, v in cells.items() if v is not None and v["V"].shape[1] >= 6}
    prog = {a: [0] * 4 for a in arms}
    for a in arms:
        for r in range(1, 5):
            n = len(glob.glob(str(R / f"outputs/ctx100_oct20/{a}/rep{r}/r*_meta.json")))
            prog[a][r - 1] = n
    if cells:
        n_eq = max(6, int(min(c["V"].shape[1] for c in cells.values()) // 2 * 2))
        for ai, (a, col) in enumerate(arms.items()):
            e = [enet(c["V"], c["yv"], c["T"], c["yt"], range(n_eq))[0]["genomic"] for (aa, r), c in cells.items() if aa == a]
            if e:
                ax.scatter(np.full(len(e), ai) + jitter(len(e), 0.06, ai), e, s=34, color=col, zorder=3); ax.hlines(np.mean(e), ai - 0.2, ai + 0.2, color=col, lw=2)
        ax.set_ylabel(f"ensemble test r at equal n = {n_eq} models")
        ax.set_title("NEW: context 30k -> 100k, 4 reps/arm (running)")
        ax.set_xticks(range(4)); ax.set_xticklabels(["none", "T top-by-val\n(30k)", "J T+trajectory\n(30k)", "X top-by-val\n(10k)"], fontsize=8)
    else:
        for ai, (a, col) in enumerate(arms.items()):
            for r in range(1, 5):
                ax.barh(ai * 5 + r, prog[a][r - 1], color=col, height=0.8, zorder=2)
        ax.axvline(30, color=INK, ls="--", lw=0.8); ax.invert_yaxis()
        ax.set_yticks([ai * 5 + 2.5 for ai in range(4)]); ax.set_yticklabels(["none", "T top-by-val\n(30k)", "J T+trajectory\n(30k)", "X top-by-val\n(10k)"], fontsize=8)
        ax.set_xlabel("models trained so far (target 30 per rep)")
        ax.set_title("NEW: context 30k -> 100k, 4 reps/arm (running)")
    DATA["ctx100_progress"] = prog
    fig.tight_layout()
    save(fig, "context_transfer.png", out)


def fig_family(out):
    fams = ["genomic", "random", "motif_ct_enriched", "evo2", "evoaug_paper2023"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    prog = {}
    for fi, f in enumerate(fams):
        for r in (1, 2, 3):
            d = R / ("outputs/archsearch_final_oct20/son55_wide" if f == "genomic" else f"outputs/archsearch_family_oct20/{f}") / f"rep{r}"
            n = len(glob.glob(str(d / "r*_arch_*_meta.json")))
            prog[f"{f}|{r}"] = n
            ax.barh(fi * 4 + r, n, color=[BLUE, TEAL, ORANGE, PURPLE, GREY][fi], height=0.8)
    ax.axvline(60, color=INK, ls="--", lw=0.8)
    ax.set_yticks([fi * 4 + 2 for fi in range(5)]); ax.set_yticklabels(fams, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlabel("models trained (target 60 per run)"); ax.set_title("Reservoir-family AutoResearch runs (3 each): progress")
    ax = axes[1]
    best = {}
    for fi, f in enumerate(fams):
        for r in (1, 2, 3):
            d = R / ("outputs/archsearch_final_oct20/son55_wide" if f == "genomic" else f"outputs/archsearch_family_oct20/{f}") / f"rep{r}"
            c = load_cell(d)
            if c is not None and c["V"].shape[1] >= 8:
                v = np.sort(c["val"])[-5:].mean()
                ax.scatter([fi + jitter(1, 0.07, fi * 3 + r)[0]], [v], s=30, color=[BLUE, TEAL, ORANGE, PURPLE, GREY][fi], zorder=3)
                best.setdefault(f, []).append(float(v))
    ax.set_xticks(range(5)); ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=7.5)
    ax.set_ylabel("mean validation r of the 5 best models"); ax.set_title("Best models per family (each on its own data)")
    DATA["family"] = dict(progress=prog, top5=best)
    fig.tight_layout()
    save(fig, "families.png", out)


def fig_menu_fam(out):
    """The same menu retrained on each reservoir family at 30k: do rankings and ensembles transfer between reservoirs?"""
    base = R / "outputs/menu_retrain_v0"
    fams = [("random", ORANGE), ("motif_ct_enriched", TEAL), ("evo2", PURPLE), ("evoaug_paper2023", GREY)]
    g = load_menu([str(base / "g30k"), str(base / "g30k_seeds12")])
    gstat = {e: dict(val=np.mean([r["val"] for r in v]), test=np.mean([r["test"] for r in v])) for e, v in g[2].items()}
    res = {}
    for f, _ in fams:
        yv, yt, ent = load_menu([str(base / f"f30k_{f}")])
        if len(ent) >= 8:
            res[f] = (yv, yt, ent)
    prog = {f: len(load_menu([str(base / f"f30k_{f}")])[2]) for f, _ in fams}
    DATA["menu_fam"] = dict(progress=prog, n_menu=len(gstat))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    ax = axes[0]
    for i, (f, col) in enumerate(fams):
        ax.barh(i, prog[f], color=col, height=0.6, zorder=2); ax.text(prog[f] + 0.5, i, f"{prog[f]} / {len(gstat)}", va="center", fontsize=8.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([f.replace("_", "\n") for f, _ in fams], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, len(gstat) * 1.3)
    ax.set_xlabel("menu architectures retrained so far (30k)"); ax.set_title("Menu retrained on each reservoir: progress")
    if res:
        ax = axes[1]
        sp = {}
        for f, col in fams:
            if f not in res:
                continue
            ent = res[f][2]
            common = sorted(set(ent) & set(gstat))
            a = [gstat[c]["test"] for c in common]; b = [np.mean([r["test"] for r in ent[c]]) for c in common]
            ax.scatter(a, b, s=22, color=col, alpha=0.8, label=f"{f.replace('_', ' ')}  rho = {spearmanr(a, b)[0]:.2f}")
            sp[f] = float(spearmanr(a, b)[0])
        ax.set_xlabel("test r when trained on genomic 30k"); ax.set_ylabel("test r when trained on the other reservoir")
        ax.legend(fontsize=7.5, loc="lower right"); ax.set_title("Do architecture rankings transfer between reservoirs?")
        DATA["menu_fam"]["spearman_vs_genomic"] = sp
        ax = axes[2]
        rows = []
        for i, (f, col) in enumerate([("genomic", BLUE)] + [x for x in fams if x[0] in res]):
            yv, yt, ent = (g if f == "genomic" else res[f])
            keys = sorted(ent)
            V = np.column_stack([np.mean([r["vp"] for r in ent[c]], 0) for c in keys])
            T = {e: np.column_stack([np.mean([r["tp"][e] for r in ent[c]], 0) for c in keys]) for e in EV}
            en = enet(V, yv, T, yt, range(V.shape[1]))[0]
            best = max(np.mean([r["test"] for r in ent[c]]) for c in keys)
            ax.bar(i - 0.2, best, 0.38, color=col, alpha=0.45, zorder=2); ax.bar(i + 0.2, en["genomic"], 0.38, color=col, zorder=2)
            rows.append(dict(res=f, best_single=float(best), ensemble=float(en["genomic"]), ood=float(en["ood"])))
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["res"].replace("_", "\n") for r in rows], fontsize=7.5)
        ax.set_ylabel("genomic-eval test r"); ax.set_title("Pale: best single model. Solid: ensemble of the menu")
        lo = min(min(r["best_single"], r["ensemble"]) for r in rows); ax.set_ylim(max(0, lo - 0.05), max(r["ensemble"] for r in rows) + 0.02)
        DATA["menu_fam"]["rows"] = rows
    else:
        for ax in axes[1:]:
            ax.axis("off"); ax.text(0.5, 0.5, "running: results appear when at least\n8 menu entries are done per reservoir", ha="center", va="center", transform=ax.transAxes, color=MUTED)
    fig.tight_layout()
    save(fig, "menu_families.png", out)


def fig_budget(out):
    root = Path("/grid/wsbs/home_norepl/christen/.claude/projects")
    EDT = dt.timezone(dt.timedelta(hours=-4))
    seen, hours = set(), collections.defaultdict(lambda: [0, 0, 0])
    for f in root.rglob("*.jsonl"):
        try:
            if f.stat().st_mtime < time.time() - 4 * 86400:
                continue
            for line in open(f, errors="replace"):
                if '"usage"' not in line:
                    continue
                o = json.loads(line)
                m = o.get("message") or {}
                u = m.get("usage")
                if not u or o.get("type") != "assistant":
                    continue
                k = (m.get("id"), o.get("requestId"))
                if k in seen:
                    continue
                seen.add(k)
                t = dt.datetime.fromisoformat(o["timestamp"].replace("Z", "+00:00")).astimezone(EDT).replace(minute=0, second=0, microsecond=0)
                tok = (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                hours[t][0] += 1; hours[t][1] += tok; hours[t][2] += u.get("output_tokens") or 0
        except Exception:
            continue
    if not hours:
        return
    hs = sorted(hours)
    hs = [h for h in hs if h >= hs[-1] - dt.timedelta(hours=60)]
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    ax.bar(hs, [hours[h][1] / 1e6 for h in hs], width=0.035, color=BLUE, label="input + output + cache-write (HPC cells)")
    ax.bar(hs, [hours[h][2] / 1e6 for h in hs], width=0.035, color=ORANGE, label="of which output")
    lim = dt.datetime(2026, 10, 20, 20, 10, tzinfo=EDT)
    if hs[0] <= lim <= hs[-1] + dt.timedelta(hours=1):
        ax.axvline(lim, color=RED, lw=1.4); ax.text(lim, ax.get_ylim()[1] * 0.93, " session limit hit 20:10\n (resets 9pm)", color=RED, fontsize=8, va="top")
    ax.set_ylabel("million tokens per hour"); ax.set_title("LLM token use by the HPC experiment cells (America/New_York)"); ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "llm_budget.png", out)
    tot = sum(h[1] for h in hours.values()); DATA["budget"] = dict(hours=len(hours), raw_tokens=int(tot), output_tokens=int(sum(h[2] for h in hours.values())), calls=int(sum(h[0] for h in hours.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/justification")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = R / a.out
    prev = out / "data.json"
    if prev.exists():  # partial runs (--only) keep the other sections
        DATA.update({k: v for k, v in json.load(open(prev)).items() if k != "generated"})
    steps = [("loop", fig_loop), ("menu", fig_menu), ("instances", fig_instances),
             ("prompt", lambda o: arm_figure(ARMS, "Prompt structure (Sonnet 5.5, genomic 30k, 30 rounds x 2)", "prompt_arms.png", "prompt_arms", o)),
             ("proposer", lambda o: arm_figure(PROP, "Proposer model (build-on-what-works prompt, genomic 30k)", "proposer_models.png", "proposer", o)),
             ("llm", fig_llm_vs_classical), ("kb", fig_kb_replicates), ("context", fig_context), ("family", fig_family), ("menufam", fig_menu_fam), ("budget", fig_budget)]
    for name, fn in steps:
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time()
        try:
            fn(out)
        except Exception as e:  # noqa: BLE001 - one broken figure must not stop the rest
            import traceback
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
        print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(DATA, open(out / "data.json", "w"), indent=1, default=float)
    print("data.json keys:", list(DATA))


if __name__ == "__main__":
    main()
