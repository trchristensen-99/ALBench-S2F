"""Genomic-reservoir ENSEMBLE performance on the three headline eval sets.

Reports Pearson r for:
  genomic   - held-out chr7+13 real sequence
  snv_delta - variant effect: corr(pred_alt - pred_ref, oracle_alt - oracle_ref),
              restricted to TRUE substitutions (Hamming == 1). The shipped battery
              is only ~91% Hamming-1; the rest are unrelated sequence pairs, and
              including them inflates the delta correlation because it starts
              measuring absolute level again rather than variant effect.
  ood       - designed high-activity sequences, outside the natural range

Ensemble = greedy-forward member selection on validation, then non-negative
ElasticNetCV on those members. Fit on val only; test is never used for selection.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/experiments")

CELLS = {
    10_000:  f"{REPO}/outputs/scaling_v2/genomic_d10000",
    30_000:  f"{REPO}/outputs/scaling_v2/genomic_d30000",
    100_000: f"{REPO}/outputs/scaling_v2/genomic_d100000",
    300_000: f"{REPO}/outputs/overnight/fresh_d300000_genomic_seed42",
}


def pear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 8 else float("nan")


def load(d):
    V, T = [], {}
    for mp in sorted(glob.glob(f"{d}/r*_meta.json")):
        try:
            j = json.load(open(mp))
        except Exception:
            continue
        if j.get("error"):
            continue
        f = mp.replace("_meta.json", ".npz")
        if not os.path.exists(f):
            continue
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if "val_pred" not in z.files:
            continue
        V.append(np.asarray(z["val_pred"], np.float64).ravel())
        for k in z.files:
            if k.startswith("test_pred_"):
                T.setdefault(k[10:], []).append(np.asarray(z[k], np.float64).ravel())
    return V, T


def stack(Vm, yv, max_k=12):
    from sklearn.linear_model import ElasticNetCV
    n, chosen, best = Vm.shape[0], [], -np.inf
    while len(chosen) < min(max_k, n):
        c, cs = None, best
        for i in range(n):
            if i not in chosen:
                s = pear(Vm[chosen + [i]].mean(0), yv)
                if s > cs:
                    c, cs = i, s
        if c is None:
            break
        chosen.append(c); best = cs
    try:
        en = ElasticNetCV(positive=True, cv=3, max_iter=5000).fit(Vm[chosen].T, yv)
        if not np.any(en.coef_ > 0):
            raise ValueError
        return chosen, (lambda M: M[chosen].T @ en.coef_ + en.intercept_)
    except Exception:
        return chosen, (lambda M: M[chosen].T.mean(1))


def main():
    from experiments.scaling_hp_search_ms import load_all_test_sets
    sets = load_all_test_sets()
    z = np.load(f"{REPO}/outputs/chr_split_cache/chr_val_ref_only.npz", allow_pickle=True)
    yv = np.asarray(z["oracle_labels"], np.float64).ravel()

    ref_s, ref_y = sets["snv_ref"]; alt_s, alt_y = sets["snv_alt"]
    ham1 = np.array([sum(a != b for a, b in zip(str(r), str(a2)))
                     for r, a2 in zip(ref_s, alt_s)]) == 1
    d_true = np.asarray(alt_y, np.float64)[ham1] - np.asarray(ref_y, np.float64)[ham1]
    print(f"  SNV pairs: {len(ham1):,} total, {ham1.sum():,} true substitutions "
          f"({100*ham1.mean():.1f}%) -- delta computed on those only\n")

    note = {}
    print("  %-9s %7s %9s %11s %9s" % ("D", "n_mod", "genomic", "snv_delta", "ood"))
    for D, d in CELLS.items():
        if not os.path.isdir(d):
            print("  %-9d %7s  <cell missing>" % (D, "-")); continue
        V, T = load(d)
        if len(V) < 3:
            print("  %-9d %7d  <too few models>" % (D, len(V))); continue
        L = max(len(v) for v in V)
        keep = [i for i, v in enumerate(V) if len(v) == L]
        Vm = np.stack([V[i] for i in keep])
        if len(yv) != Vm.shape[1]:
            # This cell validated on its OWN 10% holdout, not the shared chr-val set,
            # so greedy stacking against chr-val labels is impossible. Fall back to an
            # equal-weight average of the top-k by each model's RECORDED val score.
            # Weaker than the ElasticNet stack, and (critically) these models were
            # trained against a DIFFERENT oracle -- see the footnote printed below.
            scores = []
            for mp in sorted(glob.glob(f"{d}/r*_meta.json")):
                try:
                    j = json.load(open(mp))
                except Exception:
                    continue
                if j.get("error") or not os.path.exists(mp.replace("_meta.json", ".npz")):
                    continue
                scores.append(j.get("val_pearson") or j.get("best_val_pearson") or -9)
            scores = [scores[i] for i in keep] if len(scores) >= len(keep) else None
            if scores is None:
                print("  %-9d %7d  <no val scores>" % (D, len(keep))); continue
            topk = list(np.argsort(scores)[::-1][:8])
            apply = (lambda M, t=topk: M[t].T.mean(1))
            note[D] = "top-8 equal weight; DIFFERENT oracle + val protocol"
        else:
            _, apply = stack(Vm, yv)
        out = {}
        for name, es in (("genomic", "genomic"), ("ood", "ood")):
            if es in T and len(T[es]) >= len(keep):
                M = np.stack([T[es][i] for i in keep])
                if M.shape[1] == len(sets[es][1]):
                    out[name] = pear(apply(M), np.asarray(sets[es][1], np.float64))
        if "snv_ref" in T and "snv_alt" in T:
            R = np.stack([T["snv_ref"][i] for i in keep])
            A = np.stack([T["snv_alt"][i] for i in keep])
            if R.shape[1] == len(ham1):
                out["snv_delta"] = pear(apply(A)[ham1] - apply(R)[ham1], d_true)
        print("  %-9d %7d %9s %11s %9s" % (
            D, Vm.shape[0],
            f"{out.get('genomic', float('nan')):.4f}",
            f"{out.get('snv_delta', float('nan')):.4f}",
            f"{out.get('ood', float('nan')):.4f}"))


    for D, t in note.items():
        print(f"\n  NOTE D={D}: {t}")

if __name__ == "__main__":
    main()
