"""If a cell type has little data, what is the FASTEST way to improve it?

fs42 answered "how much data", on the random-draw assumption: pretraining on one other
cell is worth about 100,000 target-cell labels. It never asked WHICH target-cell sequences
to measure, and that is the question a collaborator with a fixed SK-N-SH budget actually
faces.

This is fully simulatable today and needs no new assay. The Tewhey corpus measures all
seven cells, so "acquiring" K target-cell labels is revealing labels we already hold, and
the chromosome holdout still protects the evaluation because the acquisition pool is drawn
from TRAIN chromosomes only.

THE STRATEGIES, one per idea worth distinguishing:

  random              the floor. Any strategy that does not beat it is noise.
  uncertainty         MC-dropout spread on the target cell. The standard answer.
  activity_normalized the same spread, residualised against predicted activity. Measured
                      on 2026-11-24, raw uncertainty partly re-ranks by activity because
                      labels are log2FC and absolute error grows with magnitude -- on the
                      K562 pools, normalising kept 86% of the uncertainty gain at ZERO
                      activity shift. Without this arm, "uncertainty won" is ambiguous.
  contrastive         largest |y_target - mean(y_other cells)| under the pretrained model:
                      the sequences the model believes behave DIFFERENTLY here. This is
                      the one with a mechanistic story, and the one aimed squarely at
                      cell-type-specific design rather than at overall fit.
  expected_activity   top predicted activity in the target. Pure exploitation, and what
                      someone would do with no framework at all.
  diverse             k-means++ over model embeddings, no label information used. The
                      control that says whether any of the above beat simply spreading
                      out.

THE ENDPOINTS. A single r on the target cell would hide the thing most worth knowing,
because a strategy can improve overall fit while making cell-type specificity worse. Every
run reports, on the target cell's held-out TEST chromosomes:

  r_all          overall Pearson, with its measurement ceiling and the gap to it
  r_high         r within the top activity tertile -- the regime a designer cares about
  r_specific     r on the DIFFERENCE between the target and a reference cell, over
                 sequences measured in both. This is the cell-type-specificity endpoint
                 and it is the one that can move the opposite way to r_all.
  prec_at_1000   of the 1,000 sequences predicted highest, the fraction truly in the top
                 1,000. A ranking endpoint, because a design campaign ranks.

Report `ceiling - r`, never `1 - r`: on these cells the ceiling is well below 1 and the
difference is larger than most of the effects here.

    python -m albench.celltype.datapoor --target-cell SK-N-SH \\
        --strategies random,uncertainty,contrastive --shots 1000,5000,20000,100000 \\
        --out outputs/celltype/dp48_SK-N-SH
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from albench.celltype import conditioning as C
from albench.celltype import data as D
from albench.celltype.fewshot import build, pearson, run_epochs

STRATEGIES = (
    "random",
    "uncertainty",
    "activity_normalized",
    "contrastive",
    "expected_activity",
    "diverse",
)


# --------------------------------------------------------------------------- scoring


def _predict(model, data, rows: np.ndarray, j: int, device: str, mc: int = 0) -> np.ndarray:
    """Two-strand mean prediction for cell ``j``. ``mc`` > 0 returns MC-dropout samples."""
    model.train() if mc else model.eval()
    reps = mc or 1
    out = np.zeros((reps, len(rows)), np.float32)
    with torch.no_grad():
        for r in range(reps):
            for s in range(0, len(rows), 1024):
                x = D.onehot_from_idx(data.idx[rows[s : s + 1024]], device)
                ci = torch.full((x.shape[0],), j, dtype=torch.long, device=device)
                p = model(x, cell_idx=ci).reshape(-1)
                p = p + model(x.flip(1).flip(2), cell_idx=ci).reshape(-1)
                out[r, s : s + 1024] = (p / 2).float().cpu().numpy()
    model.eval()
    return out if mc else out[0]


def _ceiling(y: np.ndarray, se: np.ndarray) -> float:
    var, noise = float(np.var(y)), float(np.mean(np.square(se)))
    return float(np.sqrt(max(0.0, 1 - noise / var))) if var > 0 else float("nan")


def score_panel(model, data, rows_mask, cell: str, ref_cell: str, device: str) -> dict:
    """The eval panel, not a single number. See the module docstring for why."""
    j = data.cells.index(cell)
    sel = np.flatnonzero(rows_mask)
    sel = sel[data.mask[sel, j]]
    p = _predict(model, data, sel, j, device)
    y, se = data.y[sel, j], data.se[sel, j]

    ceil = _ceiling(y, se)
    r_all = pearson(p, y)
    res = {
        "n": int(len(sel)),
        "r_all": round(r_all, 4),
        "ceiling": round(ceil, 4),
        "gap": round(ceil - r_all, 4),
    }

    # High-activity tertile: a designer's regime, and where a corpus that only taught the
    # middle of the range will show it.
    hi = y >= np.quantile(y, 2 / 3)
    if hi.sum() > 50:
        res["r_high"] = round(pearson(p[hi], y[hi]), 4)
        res["ceiling_high"] = round(_ceiling(y[hi], se[hi]), 4)

    # Ranking endpoint.
    k = min(1000, len(sel) // 4)
    if k > 50:
        res["prec_at_1000"] = round(
            float(len(np.intersect1d(np.argsort(p)[-k:], np.argsort(y)[-k:])) / k),
            4,
        )

    # Cell-type SPECIFICITY: r on the difference, over sequences measured in BOTH cells.
    # Predicting the difference is a strictly harder and more useful thing than predicting
    # either level, and it can move the opposite way to r_all.
    if ref_cell in data.cells:
        jr = data.cells.index(ref_cell)
        both = np.flatnonzero(rows_mask & data.mask[:, j] & data.mask[:, jr])
        if len(both) > 200:
            pt = _predict(model, data, both, j, device)
            pr = _predict(model, data, both, jr, device)
            dy = data.y[both, j] - data.y[both, jr]
            dse = np.sqrt(np.square(data.se[both, j]) + np.square(data.se[both, jr]))
            dp = pt - pr
            if float(np.ptp(dp)) == 0.0:
                # Not a NaN to be averaged over later: the quantity does not exist for this
                # model. Say which, so a silent nan column is never read as "hard endpoint".
                res["r_specific_unavailable"] = "model predicts both cells identically"
            else:
                res["r_specific"] = round(pearson(dp, dy), 4)
                res["ceiling_specific"] = round(_ceiling(dy, dse), 4)
                res["n_specific"] = int(len(both))
    return res


# ----------------------------------------------------------------------- acquisition


def acquire(
    strategy: str,
    model,
    data,
    avail: np.ndarray,
    k: int,
    j: int,
    other_j: list[int],
    device: str,
    seed: int,
    mc: int = 8,
) -> np.ndarray:
    """Choose which ``k`` of ``avail`` to measure in the target cell."""
    rng = np.random.default_rng(seed)
    if k >= len(avail):
        return avail
    if strategy == "random":
        return rng.choice(avail, k, replace=False)

    if strategy in ("uncertainty", "activity_normalized"):
        samples = _predict(model, data, avail, j, device, mc=mc)
        unc = samples.std(axis=0)
        if strategy == "uncertainty":
            return avail[np.argsort(unc)[-k:]]
        # Residualise against predicted activity in equal-count bins, so what is selected
        # is "uncertain FOR ITS ACTIVITY" rather than "active". Median/MAD rather than
        # mean/sd: the few genuinely uncertain points are what we are looking for and
        # would inflate an sd.
        act = samples.mean(axis=0)
        nb = max(2, min(20, len(avail) // 50))
        edges = np.quantile(act, np.linspace(0, 1, nb + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)
        b = np.clip(np.digitize(act, edges[1:-1], right=False), 0, nb - 1)
        resid = np.zeros_like(unc)
        for i in range(nb):
            m = b == i
            if m.sum() < 3:
                continue
            c = np.median(unc[m])
            mad = np.median(np.abs(unc[m] - c))
            scale = 1.4826 * mad if mad > 0 else (unc[m].std() or 1.0)
            resid[m] = (unc[m] - c) / scale
        return avail[np.argsort(resid)[-k:]]

    if strategy == "contrastive":
        pt = _predict(model, data, avail, j, device)
        others = np.stack([_predict(model, data, avail, o, device) for o in other_j])
        d = np.abs(pt - others.mean(axis=0))
        # A model that cannot tell the cells apart returns an all-equal d, and argsort of a
        # constant is index order -- so this would quietly select the last k rows in
        # chromosome order and report it as a strategy. That is how the first dp48 run
        # spent its budget. Fail instead.
        if float(np.ptp(d)) == 0.0:
            raise RuntimeError(
                "contrastive: the model predicts an identical value for the target and "
                "reference cells, so there is no contrast to rank by. An `agnostic` model "
                "ignores cell_idx entirely; use --mode onehot (cd46: onehot is the ceiling "
                "for a cell that was trained on)."
            )
        return avail[np.argsort(d)[-k:]]

    if strategy == "expected_activity":
        return avail[np.argsort(_predict(model, data, avail, j, device))[-k:]]

    if strategy == "diverse":
        # k-means++ seeding over a cheap sequence embedding. No label or model-output
        # information is used, which is the point: it is the "just spread out" control.
        from sklearn.cluster import MiniBatchKMeans

        sub = avail if len(avail) <= 50_000 else rng.choice(avail, 50_000, replace=False)
        X = data.idx[sub].astype(np.float32)
        X = X[:, :: max(1, X.shape[1] // 64)]
        km = MiniBatchKMeans(n_clusters=min(k, len(sub)), random_state=seed, n_init=3)
        lab = km.fit_predict(X)
        # One representative per cluster, nearest the centre.
        picks = []
        for c in range(km.n_clusters):
            m = np.flatnonzero(lab == c)
            if m.size:
                d = np.linalg.norm(X[m] - km.cluster_centers_[c], axis=1)
                picks.append(sub[m[int(np.argmin(d))]])
        picks = np.array(picks, dtype=int)
        if len(picks) < k:
            rest = np.setdiff1d(avail, picks)
            picks = np.concatenate([picks, rng.choice(rest, k - len(picks), replace=False)])
        return picks[:k]

    raise ValueError(f"unknown strategy {strategy!r}; have {STRATEGIES}")


# ------------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target-cell", default="SK-N-SH")
    ap.add_argument("--ref-cell", default="K562", help="reference for the specificity endpoint")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--shots", default="1000,5000,20000,100000")
    # onehot/deep, not agnostic/readout. An agnostic model has no cell input at all, so it
    # predicts one number per sequence: the specificity endpoint is identically zero (it
    # reported r_spec=nan in every cell of the first run) and `contrastive` degenerates to
    # index order. cd46 also measured onehot to be the best conditioning available for a
    # cell that was trained on -- 0.8180 genomic against agnostic's 0.7830, and 0.8144
    # against 0.2937 on the designed set -- and indistinguishable from `embed` and `text`,
    # so it is both the correct and the simplest choice here.
    ap.add_argument("--mode", default="onehot", choices=C.CONDITIONING_MODES)
    ap.add_argument("--depth", default="deep", choices=C.DEPTHS)
    ap.add_argument("--d-cond", type=int, default=32)
    ap.add_argument("--pre-epochs", type=int, default=10)
    ap.add_argument("--adapt-epochs", type=int, default=20)
    ap.add_argument("--pre-subset", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--adapt-lr", type=float, default=0.001)
    ap.add_argument("--mc", type=int, default=8, help="MC-dropout passes for uncertainty")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    data = D.load_tewhey()
    split = data.split()
    cells = list(data.cells)
    tgt = D.canonical_cell(a.target_cell)
    ref = D.canonical_cell(a.ref_cell)
    if tgt not in cells:
        raise SystemExit(f"{tgt} is not a corpus cell: {cells}")
    tj = cells.index(tgt)
    pre_cells = [c for c in cells if c != tgt]
    other_j = [cells.index(c) for c in pre_cells]

    avail = np.flatnonzero(split["train"] & data.mask[:, tj])
    shots = [int(x) for x in a.shots.split(",")]
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    # Checked here rather than at the point of use: the pretrain is ~15 minutes and there is
    # no reason to spend it on a configuration whose headline endpoint cannot be computed.
    if a.mode == "agnostic" and ("contrastive" in strategies or ref in data.cells):
        raise SystemExit(
            "--mode agnostic has no cell input, so the model predicts one number per "
            "sequence. The r_specific endpoint is then identically zero and `contrastive` "
            "has nothing to rank. Pass --mode onehot (the default), or drop both the "
            "contrastive strategy and the --ref-cell."
        )
    print(
        f"target {tgt} | ref {ref} | pretrain on {pre_cells}\n"
        f"{len(avail):,} acquirable sequences on train chromosomes | shots {shots}\n"
        f"strategies {strategies}",
        flush=True,
    )

    # ONE pretrained model, reused by every strategy and every budget. Retraining it per
    # arm would put a different pretrain seed under each strategy and the comparison would
    # carry that variance for no reason.
    t0 = time.time()
    model = build(cells, a.mode, a.depth, a.d_cond, a.seed).to(device)
    si, ki, y, _ = D.long_format(data, split["train"], pre_cells)
    if a.pre_subset and len(si) > a.pre_subset:
        p = rng.choice(len(si), a.pre_subset, replace=False)
        si, ki, y = si[p], ki[p], y[p]
    run_epochs(
        model,
        data,
        si,
        ki,
        y,
        epochs=a.pre_epochs,
        batch=a.batch_size,
        lr=a.lr,
        seed=a.seed,
        device=device,
    )
    base = score_panel(model, data, split["test"], tgt, ref, device)
    print(f"\nzero-shot on {tgt} ({time.time() - t0:.0f}s): {json.dumps(base)}", flush=True)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    results = [{"strategy": "none", "shots": 0, **base}]
    for strat in strategies:
        for k in shots:
            model.load_state_dict(state)
            t1 = time.time()
            pick = acquire(strat, model, data, avail, k, tj, other_j, device, a.seed, a.mc)
            sel_s = time.time() - t1
            model.load_state_dict(state)  # selection used MC-dropout; restore cleanly
            ki2 = np.full(len(pick), tj, dtype=np.int64)
            run_epochs(
                model,
                data,
                pick,
                ki2,
                data.y[pick, tj].astype(np.float32),
                epochs=a.adapt_epochs,
                batch=min(a.batch_size, max(16, k // 4)),
                lr=a.adapt_lr,
                seed=a.seed,
                device=device,
            )
            res = score_panel(model, data, split["test"], tgt, ref, device)
            # What the strategy BOUGHT, not just how it scored: the activity profile of
            # the acquired set is half the explanation for any difference between arms.
            acq_y = data.y[pick, tj]
            res.update(
                acquired_mean=round(float(acq_y.mean()), 4),
                acquired_sd=round(float(acq_y.std()), 4),
                pool_mean=round(float(data.y[avail, tj].mean()), 4),
                select_seconds=round(sel_s, 1),
            )
            results.append({"strategy": strat, "shots": k, **res})
            print(
                "  %-20s K=%-7d r_all=%.4f (gap %.4f)  r_spec=%s  prec@1k=%s  acq_mean=%.3f"
                % (
                    strat,
                    k,
                    res["r_all"],
                    res["gap"],
                    res.get("r_specific"),
                    res.get("prec_at_1000"),
                    res["acquired_mean"],
                ),
                flush=True,
            )
            with open(os.path.join(a.out, "curve.json"), "w") as fh:
                json.dump(
                    {"target": tgt, "ref": ref, "args": vars(a), "results": results}, fh, indent=1
                )

    print("\n=== %s: r_all / r_specific by strategy and budget" % tgt)
    hdr = "%-20s" % "strategy" + "".join("%18s" % f"K={k}" for k in shots)
    print(hdr)
    for strat in strategies:
        row = "%-20s" % strat
        for k in shots:
            m = [x for x in results if x["strategy"] == strat and x["shots"] == k]
            row += "%18s" % (
                f"{m[0]['r_all']:.3f}/{m[0].get('r_specific', float('nan')):.3f}" if m else "-"
            )
        print(row)


if __name__ == "__main__":
    main()
