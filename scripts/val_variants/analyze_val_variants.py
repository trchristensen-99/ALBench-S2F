#!/usr/bin/env python
"""Analysis for the validation-variant experiment (design: docs in scripts/program_oct22/register_val_variants.py).

For every pool and every training arm (the validation set the menu was trained/early-stopped against) it reports

  (a) ENSEMBLE   test Pearson r on each of the 18 evaluation sets (+ SNV delta) of the validation-ensembled result: non-negative
                 ElasticNet (scratchpad/ens_util.enet_all_pos) fitted on THAT arm's validation predictions. With --cross, also the
                 weights fitted on each OTHER validation set the models were scored on (needs runs made with --extra_val), which
                 separates "what the weights were fitted on" from "what early stopping selected" at fixed training.
  (b) RANKING    Spearman across the 33 architectures between a validation score and the per-set test r, with a bootstrap over
                 architectures. The ceiling is the same statistic between two seeds of the SAME validation set (training noise).
  (c) EARLY STOP epochs trained, fraction stopped by patience, and the paired (same architecture, same init seed) change in single-model
                 test r relative to the baseline arm.

Baseline arm V0 = the existing default-validation runs outputs/menu_retrain_v0/ind30k_{pool}_r1 (3 seeds: r00,r01,r02), which also give the
noise floor (SD of the ensemble r over seeds). V0x repeats it with --extra_val; V0c is the same genomic validation sequences relabelled by oracle_v3
(the stored chr_val labels are a different oracle), the fair baseline for V1-V3. Everything degrades gracefully when an arm has not finished.

usage: analyze_val_variants.py [--pools genomic random motif_ct_enriched] [--arms V1 V2 V3 V0x] [--cross] [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))
MR = REPO / "outputs/menu_retrain_v0"  # --root overrides (tests)


def pear(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


class Arm:
    """All models of one result directory (several seeds = rounds)."""

    def __init__(self, d: Path):
        self.d = d
        self.labels = np.load(d / "labels.npz")
        self.sets = [k[len("oracle_") :] for k in self.labels.files if k.startswith("oracle_")]
        self.models: dict[tuple[int, int], tuple[dict, str]] = {}
        for f in sorted(glob.glob(str(d / "r*_menu_*_meta.json"))):
            m = json.load(open(f))
            self.models[(int(m["round"]), int(m["menu_index"]))] = (
                m,
                f.replace("_meta.json", ".npz"),
            )
        self.rounds = sorted({r for r, _ in self.models})
        self.extra = sorted(
            k[len("val_labels_") :] for k in self.labels.files if k.startswith("val_labels_")
        )

    def idxs(self, rnd):
        return sorted(i for r, i in self.models if r == rnd)

    def preds(self, rnd, idxs=None):
        """-> {key: (n, N)} for val_pred, valpred_*, test_pred_*; models that failed are simply absent."""
        idxs = idxs if idxs is not None else self.idxs(rnd)
        cols = [np.load(self.models[(rnd, i)][1]) for i in idxs]
        keys = [k for k in cols[0].files if k != "test_pred"]
        return idxs, {k: np.stack([np.asarray(c[k], np.float64) for c in cols], 1) for k in keys}

    def val_labels(self, name=None):
        return np.asarray(
            self.labels["val_labels" if name is None else f"val_labels_{name}"], np.float64
        )

    def test_r(self, P):
        """per-set test Pearson for every model: {set: (N,)}"""
        out = {}
        for s in self.sets:
            y = np.asarray(self.labels[f"oracle_{s}"], np.float64)
            out[s] = np.array(
                [pear(P[f"test_pred_{s}"][:, j], y) for j in range(P[f"test_pred_{s}"].shape[1])]
            )
        return out


def ensemble(arm: Arm, rnd: int, val_name=None, n_fit=8000, seed=0):
    """ElasticNet-ensemble test r per set (and SNV delta) with weights fitted on the chosen validation set."""
    from ens_util import enet_all_pos

    idxs, P = arm.preds(rnd)
    Vk = "val_pred" if val_name is None else f"valpred_{val_name}"
    yv = arm.val_labels(val_name)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(yv), min(n_fit, len(yv)), replace=False))
    T = {s: P[f"test_pred_{s}"] for s in arm.sets}
    pred, n_kept, n_drop = enet_all_pos(P[Vk], yv, T, sub)
    res = {s: pear(pred[s], arm.labels[f"oracle_{s}"]) for s in arm.sets}
    if "snv_ref" in pred and "snv_alt" in pred:
        res["snv_delta"] = pear(
            pred["snv_alt"] - pred["snv_ref"],
            np.asarray(arm.labels["oracle_snv_alt"], float)
            - np.asarray(arm.labels["oracle_snv_ref"], float),
        )
    res["_n_kept"], res["_n_models"] = n_kept, len(idxs)
    return res


def mean18(res):
    v = [res[s] for s in res if not s.startswith("_") and s != "snv_delta"]
    return float(np.mean(v))


def boot_spearman_diff(x1, x2, y, n_boot=1000, seed=0):
    """Spearman(x1,y) - Spearman(x2,y) over architectures, bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d0 = spearmanr(x1, y)[0] - spearmanr(x2, y)[0]
    ds = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if len(set(y[b])) < 3:
            continue
        ds.append(spearmanr(x1[b], y[b])[0] - spearmanr(x2[b], y[b])[0])
    return d0, tuple(np.nanpercentile(ds, [2.5, 97.5]))


def boot_mean_diff(va, vb, TR, n_boot=400, seed=0):
    """Mean over evaluation sets of [Spearman(va, test r) - Spearman(vb, test r)] across architectures, with a bootstrap CI over architectures.
    va and vb are validation scores of the SAME trained models (cross-scored), so training noise cancels in the difference."""
    rng = np.random.default_rng(seed)
    n, S = TR.shape

    def stat(ix):
        return float(
            np.mean(
                [
                    spearmanr(va[ix], TR[ix, s])[0] - spearmanr(vb[ix], TR[ix, s])[0]
                    for s in range(S)
                ]
            )
        )

    d0 = stat(np.arange(n))
    ds = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    return d0, tuple(np.nanpercentile(ds, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["genomic", "random", "motif_ct_enriched"])
    ap.add_argument("--arms", nargs="*", default=["V0x", "V0c", "V1", "V2", "V3"])
    ap.add_argument(
        "--cross",
        action="store_true",
        help="also fit the ensemble on every extra validation set (needs --extra_val runs)",
    )
    ap.add_argument("--out", default=str(REPO / "outputs/val_variants/analysis"))
    ap.add_argument(
        "--root",
        default=str(MR),
        help="directory holding ind30k_{pool}_r1 (baseline) and vv30k_{arm}_{pool}_r1",
    )
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R: dict = {}
    for pool in args.pools:
        base_dir = root / f"ind30k_{pool}_r1"
        arms = {"V0": base_dir}
        for a in args.arms:
            arms[a] = root / f"vv30k_{a}_{pool}_r1"
        arms = {
            k: Arm(v)
            for k, v in arms.items()
            if (v / "labels.npz").exists() and glob.glob(str(v / "r*_meta.json"))
        }
        if "V0" not in arms:
            print(f"[{pool}] no baseline")
            continue
        print(
            f"\n===== pool {pool}: arms { {k: (len(v.rounds), len(v.models)) for k, v in arms.items()} } (rounds, models)"
        )
        pr = R.setdefault(pool, {})
        # ---------------- (a) validation-ensembled test r ----------------
        ens = {}
        for name, arm in arms.items():
            ens[name] = {}
            for rnd in arm.rounds:
                if len(arm.idxs(rnd)) < 20:
                    continue
                ens[name][rnd] = ensemble(arm, rnd)
        sets = arms["V0"].sets + ["snv_delta"]
        base_rounds = sorted(ens["V0"])
        sd = {
            s: float(np.std([ens["V0"][r][s] for r in base_rounds], ddof=1))
            if len(base_rounds) > 1
            else float("nan")
            for s in sets
        }
        pr["ensemble"] = {n: {str(r): v for r, v in e.items()} for n, e in ens.items()}
        pr["noise_sd_V0_seeds"] = sd
        print("(a) ensemble test r (mean over the 18 sets), by arm and seed round:")
        for n, e in ens.items():
            print(
                f"    {n:4s} "
                + "  ".join(
                    f"r{r:02d}={mean18(v):.4f}(k={v['_n_kept']}/{v['_n_models']})"
                    for r, v in sorted(e.items())
                )
            )
        if len(base_rounds) > 1:
            print(
                f"    V0 seed SD of mean18 = {np.std([mean18(ens['V0'][r]) for r in base_rounds], ddof=1):.4f}"
            )
        ref = ens["V0"][0] if 0 in ens["V0"] else ens["V0"][base_rounds[0]]
        print(
            f"    per-set difference to V0 round 0 (arm round 0; in units of the V0 seed SD where available):"
        )
        print("    " + f"{'set':14s}" + "".join(f"{n:>14s}" for n in arms if n != "V0"))
        for s in sets:
            row = f"    {s:14s}"
            for n in arms:
                if n == "V0":
                    continue
                if 0 not in ens.get(n, {}):
                    row += f"{'-':>14s}"
                    continue
                dlt = ens[n][0][s] - ref[s]
                z = dlt / sd[s] if sd[s] and np.isfinite(sd[s]) and sd[s] > 0 else float("nan")
                row += f"{dlt:+9.4f}({z:+.1f})" if np.isfinite(z) else f"{dlt:+9.4f}(  ) "
            print(row)
        # ---------------- (a') cross: weights fitted on other validation sets, training fixed ----------------
        if args.cross:
            cross = {}
            for n, arm in arms.items():
                if not arm.extra or 0 not in arm.rounds:
                    continue
                cross[n] = {v: mean18(ensemble(arm, 0, val_name=v)) for v in arm.extra}
            pr["cross_weighting_mean18"] = cross
            print(
                "(a') mean18 test r for models trained on <row>, ensemble weights fitted on validation set <column>:"
            )
            for n, d in cross.items():
                print(f"    trained on {n:4s} " + "  ".join(f"{v}={x:.4f}" for v, x in d.items()))
        # ---------------- (b) ranking of architectures ----------------
        print(
            "(b) Spearman across architectures: validation score vs test r (mean over sets; own validation set unless noted)"
        )
        rank = {}
        for n, arm in arms.items():
            if 0 not in arm.rounds:
                continue
            idxs, P = arm.preds(0)
            tr = arm.test_r(P)
            cols = {"own": ("val_pred", None)}
            for v in arm.extra:
                cols[f"on_{v}"] = (f"valpred_{v}", v)
            vrs = {}
            for cn, (k, vn) in cols.items():
                yv = arm.val_labels(vn)
                vr = np.array([pear(P[k][:, j], yv) for j in range(P[k].shape[1])])
                vrs[cn] = vr
                per = {s: float(spearmanr(vr, tr[s])[0]) for s in arm.sets}
                rank[f"{n}|{cn}"] = {
                    "per_set": per,
                    "mean18": float(np.mean(list(per.values()))),
                    "val_r_mean": float(vr.mean()),
                }
            if (
                "on_V0" in vrs
            ):  # same trained models, scored on every validation set: paired difference against V0 (training noise cancels)
                TR = np.stack([tr[s] for s in arm.sets], 1)
                for cn in vrs:
                    if cn in ("own", "on_V0"):
                        continue
                    d0, ci = boot_mean_diff(vrs[cn], vrs["on_V0"], TR)
                    rank[f"{n}|{cn}_minus_on_V0"] = {
                        "mean18": float(d0),
                        "ci95": [float(ci[0]), float(ci[1])],
                    }
        # ceiling: seed r00 vs r01 test-r ranking agreement, and V0 val score (seed 0) vs test r of seed 1
        arm0 = arms["V0"]
        if len(arm0.rounds) > 1:
            i0, P0 = arm0.preds(0)
            i1, P1 = arm0.preds(1)
            common = sorted(set(i0) & set(i1))
            c0 = [i0.index(i) for i in common]
            c1 = [i1.index(i) for i in common]
            t0, t1 = arm0.test_r(P0), arm0.test_r(P1)
            ceil = {
                s: float(spearmanr(np.array(t0[s])[c0], np.array(t1[s])[c1])[0]) for s in arm0.sets
            }
            yv = arm0.val_labels()
            v0 = np.array([pear(P0["val_pred"][:, j], yv) for j in c0])
            xs = {s: float(spearmanr(v0, np.array(t1[s])[c1])[0]) for s in arm0.sets}
            rank["V0|ceiling_test_seed0_vs_seed1"] = {
                "per_set": ceil,
                "mean18": float(np.mean(list(ceil.values()))),
            }
            rank["V0|own_val_seed0_vs_test_seed1"] = {
                "per_set": xs,
                "mean18": float(np.mean(list(xs.values()))),
            }
        pr["ranking"] = rank
        for k, v in rank.items():
            if "ci95" in v:
                print(
                    f"    {k:34s} mean18 delta Spearman = {v['mean18']:+.3f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                )
            else:
                print(
                    f"    {k:34s} mean18 Spearman = {v['mean18']:.3f}"
                    + (f"   (mean val r {v['val_r_mean']:.4f})" if "val_r_mean" in v else "")
                )
        # bootstrap: does any variant's own validation rank architectures better than V0's (per eval set family)?
        for n in arms:
            if n == "V0" or 0 not in arms[n].rounds or 0 not in arms["V0"].rounds:
                continue
            ia, Pa = arms[n].preds(0)
            ib, Pb = arms["V0"].preds(0)
            common = sorted(set(ia) & set(ib))
            ca, cb = [ia.index(i) for i in common], [ib.index(i) for i in common]
            va = np.array([pear(Pa["val_pred"][:, j], arms[n].val_labels()) for j in ca])
            vb = np.array([pear(Pb["val_pred"][:, j], arms["V0"].val_labels()) for j in cb])
            tb = arms["V0"].test_r(Pb)
            diffs = {}
            for s in arms["V0"].sets:
                d0, ci = boot_spearman_diff(va, vb, np.array(tb[s])[cb])
                diffs[s] = {"delta_spearman": float(d0), "ci95": [float(ci[0]), float(ci[1])]}
            pr.setdefault("ranking_vs_V0", {})[n] = diffs
            sig = [s for s, d in diffs.items() if d["ci95"][0] > 0 or d["ci95"][1] < 0]
            print(
                f"    {n} own-val vs V0 own-val ranking of V0-trained models' test r: mean delta {np.mean([d['delta_spearman'] for d in diffs.values()]):+.3f}; sets with CI excluding 0: {sig}"
            )
        # ---------------- (c) early stopping ----------------
        print(
            "(c) early stopping and paired single-model effect (arm minus V0, same architecture, same init seed, round 0):"
        )
        es = {}
        for n, arm in arms.items():
            if 0 not in arm.rounds:
                continue
            ms = [arm.models[(0, i)][0] for i in arm.idxs(0)]
            ep = np.array([m["epochs_trained"] or 0 for m in ms], float)
            planned = np.array([m["train_config"]["epochs"] for m in ms], float)
            es[n] = {
                "epochs_mean": float(ep.mean()),
                "frac_stopped_by_patience": float((ep < planned).mean()),
                "val_r_mean": float(np.mean([m["val_pearson"] for m in ms])),
                "test_genomic_r_mean": float(np.mean([m["test_oracle_pearson"] for m in ms])),
            }
            print(
                f"    {n:4s} epochs {ep.mean():5.1f} of {planned.mean():5.1f} planned; stopped by patience {100 * (ep < planned).mean():3.0f}%; own-val r {es[n]['val_r_mean']:.4f}; genomic-test r {es[n]['test_genomic_r_mean']:.4f}"
            )
        for n, arm in arms.items():
            if n == "V0" or 0 not in arm.rounds or 0 not in arms["V0"].rounds:
                continue
            common = sorted(set(arm.idxs(0)) & set(arms["V0"].idxs(0)))
            ia, Pa = arm.preds(0, common)
            ib, Pb = arms["V0"].preds(0, common)
            ta, tb = arm.test_r(Pa), arms["V0"].test_r(Pb)
            dm = {s: float(np.mean(ta[s] - tb[s])) for s in arm.sets}
            dpe = np.array(
                [arm.models[(0, i)][0]["epochs_trained"] or 0 for i in common]
            ) - np.array([arms["V0"].models[(0, i)][0]["epochs_trained"] or 0 for i in common])
            es[n]["paired_delta_test_r"] = dm
            es[n]["paired_delta_mean18"] = float(np.mean(list(dm.values())))
            es[n]["delta_epochs_mean"] = float(dpe.mean())
            print(
                f"    {n:4s} paired delta single-model test r (mean over 18 sets) {es[n]['paired_delta_mean18']:+.4f}; delta epochs {dpe.mean():+.1f}; most negative sets: "
                + ", ".join(
                    f"{s} {v:+.3f}" for s, v in sorted(dm.items(), key=lambda kv: kv[1])[:3]
                )
            )
        pr["early_stopping"] = es
    # ---------------- (d) does the validation choice change the RESERVOIR COMPARISON? ----------------
    # The decision we actually take is "which pool trains better", so report each pool contrast under every arm and the
    # difference of differences relative to V0c (genomic validation relabelled by oracle_v3, the fair baseline) and to the status quo V0x.
    pools = [p for p in args.pools if p in R and "ensemble" in R[p]]
    if len(pools) > 1:
        arm_names = sorted({n for p in pools for n in R[p]["ensemble"] if n != "V0" or True})

        def e0(p, n):
            return R[p]["ensemble"].get(n, {}).get("0")

        contrasts = {}
        print(
            "\n(d) pool contrasts of the validation-ensembled mean test r over the 18 sets (round 0), by validation arm:"
        )
        pairs = [(a, b) for i, a in enumerate(pools) for b in pools[:i]]
        for a, b in pairs:
            row = {}
            for n in arm_names:
                ea, eb = e0(a, n), e0(b, n)
                if ea and eb:
                    row[n] = {
                        "mean18": mean18(ea) - mean18(eb),
                        "per_set": {k: ea[k] - eb[k] for k in ea if not k.startswith("_")},
                    }
            contrasts[f"{a}-{b}"] = row
            base = row.get("V0c") or row.get("V0x") or row.get("V0")
            print(
                f"    {a} minus {b}: "
                + "  ".join(f"{n}={v['mean18']:+.4f}" for n, v in row.items())
            )
            if base:
                dod = {n: v["mean18"] - base["mean18"] for n, v in row.items() if v is not base}
                print(
                    f"        difference of differences vs {'V0c' if row.get('V0c') else 'baseline'}: "
                    + "  ".join(f"{n}={d:+.4f}" for n, d in dod.items())
                )
        R["_pool_contrasts"] = contrasts
    (out / "results.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
