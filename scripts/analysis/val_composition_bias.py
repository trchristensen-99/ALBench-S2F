#!/usr/bin/env python
"""Does selecting on V0c's composition change what we select, and what it is worth?

Every choice in this study is made on the oracle_v3-labelled validation asset `val_v0c`: the stack
weights, the per-size recipe, the menu's K, the best-of-run pick, the loop's own accept/reject. That
asset is the chr19/21/X genomic validation split, and the Gosai audit (2026-11-10) measured it as
**+0.18 log2FC more active and +4.7 GC points richer than the training chromosomes**, while the
chr7/chr13 test split matches training. So every one of those choices is made on a set that is not
drawn like the data the models are trained on, nor like the data they are judged on. A reviewer will
ask whether the conclusions survive an unbiased selection set, and nothing in the program has tested
it.

This answers it without training anything: the members' validation predictions are already saved
next to their test predictions, so the selection can simply be re-run on a reweighted validation set
and scored on the same (untouched, full) test sets.

The reweighting is a 2-D stratified resample. Training sequences are binned into deciles of GC x
deciles of oracle activity; validation rows are then drawn with replacement so that their joint
occupancy matches the training one. Bins the validation set cannot fill are reported as dropped
mass rather than silently renormalised away.

**The control is the point.** A matched resample differs from the status quo in two ways at once --
it is matched, and it is a resample with replacement, which costs effective sample size. So every
matched draw is paired with a uniform resample of identical size and the comparison is
matched-vs-uniform, not matched-vs-full. Without that control a loss of effective n would read as an
effect of composition.

    python scripts/analysis/val_composition_bias.py [--size 30k] [--draws 5] [--out ...]    (cpuq)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.ensemble.stack import fit_zscored_stack  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "analysis"))
from compare_legnet_vs_loop import MR, SETS, TUNED, load, pearson  # noqa: E402

VAL_ASSET = "/grid/koo/home/shared/albench/data/val_V0c.npz"
TRAIN_CACHE = "outputs/chr_split_cache/chr_train_ref_only.npz"
NB = 10  # deciles on each axis; 100 cells over ~315k training rows is ~3k per cell
HEAD = ("genomic", "ood", "random_32k", "gc_sweep", "dinuc_shuffle")


def gc_fraction(seqs: np.ndarray) -> np.ndarray:
    """GC per sequence, vectorised over the byte view -- a Python loop over 315k strings is minutes."""
    n = len(seqs)
    width = max(len(s) for s in seqs)
    arr = np.frombuffer("".join(s.ljust(width) for s in seqs).encode(), dtype="S1").reshape(
        n, width
    )
    gc = ((arr == b"G") | (arr == b"C")).sum(1)
    valid = (arr != b" ").sum(1)
    return gc / np.maximum(valid, 1)


def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)


def matched_sampler(
    train_gc, train_act, val_gc, val_act
) -> tuple[np.ndarray, list[list[int]], float]:
    """Target cell probabilities from training, and the validation rows available in each cell."""
    ge = np.quantile(train_gc, np.linspace(0, 1, NB + 1))
    ae = np.quantile(train_act, np.linspace(0, 1, NB + 1))
    ge[0], ge[-1], ae[0], ae[-1] = -np.inf, np.inf, -np.inf, np.inf
    tcell = bin_index(train_gc, ge) * NB + bin_index(train_act, ae)
    vcell = bin_index(val_gc, ge) * NB + bin_index(val_act, ae)
    p = np.bincount(tcell, minlength=NB * NB).astype(np.float64)
    p /= p.sum()
    rows = [np.flatnonzero(vcell == c) for c in range(NB * NB)]
    empty = np.array([len(r) == 0 for r in rows])
    dropped = float(p[empty].sum())  # training mass the validation set simply cannot represent
    p = np.where(empty, 0.0, p)
    p /= p.sum()
    return p, rows, dropped


def draw(p, rows, n, rng) -> np.ndarray:
    cells = rng.choice(len(p), size=n, p=p)
    return np.array([rows[c][rng.integers(len(rows[c]))] for c in cells])


def scores(ms: list, yv: np.ndarray, y: dict, idx: np.ndarray | None) -> dict:
    """Fit the stack on the given validation rows; score on the full, untouched test sets."""
    V = np.stack([m[0] for m in ms])
    st = fit_zscored_stack(V if idx is None else V[:, idx], yv if idx is None else yv[idx])
    out = {s: pearson(st.predict(np.stack([m[1][s] for m in ms])), y[s]) for s in SETS}
    out["effective_members"] = float(st.coef.sum() ** 2 / max((st.coef**2).sum(), 1e-12))
    out["top_member"] = int(np.argmax(st.coef))
    out["coef"] = st.coef / max(st.coef.sum(), 1e-12)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="30k", choices=["10k", "30k", "100k"])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", default="outputs/justification_oct22")
    ap.add_argument(
        "--dirs",
        default="",
        help="comma-separated model directories to pool instead of the tuned bank. The bank is 20 "
        "seeds of one recipe and so the least composition-sensitive set there is; a heterogeneous "
        "set (the reference bracket, a loop run, the two pooled) is the case worth testing.",
    )
    ap.add_argument("--pattern", default="r*_menu_*.npz")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dirs = [d for d in a.dirs.split(",") if d] or [f"{MR}/bench38_ln_{TUNED[a.size]}_{a.size}_c20"]
    parts = [load(d, a.pattern) for d in dirs]
    assert all(p for p in parts), f"nothing loaded from {[d for d, p in zip(dirs, parts) if not p]}"
    yv, y = parts[0]["yv"], parts[0]["y"]
    for p in parts[1:]:  # pooling only makes sense if every part was scored on the same rows
        assert np.allclose(p["yv"], yv), "the directories do not share a validation set"
    ms = [m for p in parts for m in p["ms"]]
    assert len(ms) > 1, f"only {len(ms)} usable models"
    print(f"pooled {len(ms)} models from {len(dirs)} directory(ies)")

    v = np.load(VAL_ASSET, allow_pickle=True)
    t = np.load(TRAIN_CACHE, allow_pickle=True)
    # the saved val_pred columns are in the asset's own order; if that ever stops holding, every
    # number below is quietly wrong, so check it rather than assume it
    assert len(v["sequences"]) == len(yv), f"{len(v['sequences'])} sequences vs {len(yv)} labels"
    assert np.allclose(v["oracle_labels"].astype(np.float64), yv, atol=1e-5), (
        "val_labels in labels.npz do not match the val_V0c asset -- the columns are not aligned"
    )

    val_gc, train_gc = gc_fraction(v["sequences"]), gc_fraction(t["sequences"])
    val_act = yv
    train_act = t["oracle_labels"].astype(np.float64)
    print(
        f"composition  GC train {train_gc.mean():.4f} val {val_gc.mean():.4f} "
        f"(+{100 * (val_gc.mean() - train_gc.mean()):.2f} points) | "
        f"activity train {train_act.mean():.4f} val {val_act.mean():.4f} "
        f"(+{val_act.mean() - train_act.mean():.4f})"
    )

    p, rows, dropped = matched_sampler(train_gc, train_act, val_gc, val_act)
    print(
        f"stratified on {NB}x{NB} cells; training mass validation cannot represent: {dropped:.3%}"
    )

    full = scores(ms, yv, y, None)
    res = {
        "size": a.size,
        "n_models": len(ms),
        "n_val": int(len(yv)),
        "dropped_mass": dropped,
        "gc_gap_points": float(100 * (val_gc.mean() - train_gc.mean())),
        "act_gap": float(val_act.mean() - train_act.mean()),
        "full": {k: full[k] for k in full if k != "coef"},
        "matched": [],
        "uniform": [],
    }
    rng = np.random.default_rng(0)
    for d in range(a.draws):
        mi = draw(p, rows, len(yv), rng)
        ui = rng.integers(0, len(yv), size=len(yv))  # the control: resampled, but not matched
        for key, idx in (("matched", mi), ("uniform", ui)):
            s = scores(ms, yv, y, idx)
            s["spearman_vs_full"] = pearson(
                np.argsort(np.argsort(full["coef"])), np.argsort(np.argsort(s["coef"]))
            )
            res[key].append({k: s[k] for k in s if k != "coef"})
        print(
            f"  draw {d}: matched genomic {res['matched'][-1]['genomic']:.4f}  uniform {res['uniform'][-1]['genomic']:.4f}"
        )

    print(f"\n{'set':<16}{'full':>9}{'uniform':>12}{'matched':>12}{'matched-uniform':>18}")
    for s in HEAD + ("snv_ref",):
        u = np.array([r[s] for r in res["uniform"]])
        m = np.array([r[s] for r in res["matched"]])
        print(
            f"{s:<16}{full[s]:>9.4f}{u.mean():>12.4f}{m.mean():>12.4f}"
            f"{m.mean() - u.mean():>+13.4f} +- {m.std(ddof=1) + u.std(ddof=1):.4f}"
        )
    ub = np.mean([r["top_member"] == full["top_member"] for r in res["uniform"]])
    mb = np.mean([r["top_member"] == full["top_member"] for r in res["matched"]])
    print(
        f"\ntop-weighted member unchanged: uniform {ub:.0%} of draws, matched {mb:.0%} | "
        f"effective members full {full['effective_members']:.1f} "
        f"uniform {np.mean([r['effective_members'] for r in res['uniform']]):.1f} "
        f"matched {np.mean([r['effective_members'] for r in res['matched']]):.1f}"
    )

    os.makedirs(a.out, exist_ok=True)
    path = f"{a.out}/val_composition_bias_{a.size}{a.tag}.json"
    json.dump(res, open(path, "w"), indent=1, default=float)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
