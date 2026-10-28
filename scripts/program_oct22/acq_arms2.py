"""Acquisition experiment, second set of arms: 30k genomic + 70k selected from the 300k candidate pool by criteria beyond mean / SD, and the
"30k genomic + 70k from one reservoir" arms. Everything is computed from stored predictions, so it needs no GPU (run it on cpuq).

Setting is the one of acq_dryrun.py: base = genomic r1 30k; candidates = 100k each of random, motif_ct_enriched, evoaug_paper2023 (replicate 1),
labelled by oracle_v3, so "labelling" a selected sequence is a lookup; selection models = the menu retrained on the 30k base (outputs/menu_retrain_v0/
acq30k_genomic_r1, predictions on the candidates stored as `poolpred`). The three configurations of the skip list are left out of the selection ensemble.

Criteria (P = models x candidates, mu / sd = mean / SD over models; n = 70,000):
  resid    uncertainty excess over the trend at that activity (the library's uncertainty_activity_normalised: SD standardised by the median / MAD
           of the SD inside 20 equal-count bins of the predicted activity); plain SD is largely a function of activity
  binned   five equal-count bins of mu, highest sd inside each bin: uncertainty spread over the activity range
  diverse  1,000 k-means clusters of 4-mer frequency vectors, equal numbers at random from each cluster: diversity, no model
  stdiv    the 140k highest-sd candidates, then the cluster-balanced 70k of those: uncertainty first, then diversity
  badge    k-means++ in the space of centred member predictions (x_i = P[:, i] - mu_i, so |x_i|^2 = M sd_i^2): picks points that are far from
           everything chosen so far AND have large disagreement, the structure of BADGE's gradient embedding without needing gradients
  oracle   the 70k with the largest |label - ensemble mean| on the candidates (uses the labels: an unrealisable CEILING for error-driven selection)
  strat2d  baseline: 15 strata (5 equal-count bins of predicted activity x 3 GC terciles), equal numbers at random from each
  neyman   composition design: 45 strata (source reservoir x 5 activity bins x 3 GC terciles), allocation proportional to stratum size x mean member SD,
           random inside each stratum (Neyman allocation)
  powersd  stochastic uncertainty sampling: 70k drawn without replacement with probability proportional to SD squared (PowerBALD-like, not top-k)
Reservoir arms: res_<name>_d100000.npz for the registry pools that already exist (the others are written by slice_nested_new.py).

    python scripts/program_oct22/acq_arms2.py            # idempotent
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

R = Path(os.environ.get("ALBENCH_ROOT", "/grid/wsbs/home_norepl/christen/ALBench-S2F"))
GP = R / "outputs/grid_pools"
REPS, ACQ = GP / "k562_reps", GP / "k562_acq"
BASE_DIR = R / "outputs/menu_retrain_v0/acq30k_genomic_r1"
MENU = R / "outputs/menu_retrain_v0/menu_v0.json"
SKIP = {
    "son55_rep1_r00_arch_00",
    "son45_rep1_r11_arch_15",
    "son55_rep2_r15_arch_31",
}
N_ADD, N_CLUSTERS = 70_000, 1_000
RESERVOIR_ARMS = {  # name -> (pool file of the 100k pool, first 70k used)
    "random": REPS / "random_r1_d100000.npz",
    "motif_ct_enriched": REPS / "motif_ct_enriched_r1_d100000.npz",
    "evoaug_paper2023": REPS / "evoaug_paper2023_r1_d100000.npz",
    "motif_shared_core": REPS / "motif_shared_core_r1_d100000.npz",
    "zoonomia_orthologs": GP / "k562_nested/zoonomia_orthologs_d100000.npz",
}
log = lambda *a: print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)  # noqa: E731


def save(path, seqs, labels, strategy):
    tmp = str(path).replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(
        tmp,
        sequences=np.asarray(seqs, dtype=object),
        oracle_labels=np.asarray(labels, np.float32),
        oracle_id="oracle_v3",
        strategy=strategy,
        seed=np.int64(0),
    )
    os.replace(tmp, path)


def load_scores():
    ids = [e["id"] for e in json.load(open(MENU))]
    P, used = [], []
    for f in sorted(glob.glob(f"{BASE_DIR}/r00_menu_*_meta.json")):
        m = json.load(open(f))
        z = np.load(f.replace("_meta.json", ".npz"))
        if "poolpred" not in z.files or ids[m["menu_index"]] in SKIP:
            continue
        P.append(z["poolpred"].astype(np.float32))
        used.append(ids[m["menu_index"]])
    return np.stack(P), used


def kmer_features(seqs, k=4):
    """Normalised k-mer frequency vectors (n, 4**k)."""
    lut = np.full(256, 4, np.uint8)
    for i, c in enumerate("ACGT"):
        lut[ord(c)] = i
    X = np.zeros((len(seqs), 4**k), np.float32)
    for r, s in enumerate(seqs):
        a = lut[np.frombuffer(s.encode(), np.uint8)]
        ok = a < 4
        code = np.zeros(len(a) - k + 1, np.int64)
        valid = np.ones(len(a) - k + 1, bool)
        for j in range(k):
            code = code * 4 + a[j : len(a) - k + 1 + j]
            valid &= ok[j : len(a) - k + 1 + j]
        X[r] = np.bincount(code[valid], minlength=4**k)[: 4**k]
    return X / np.maximum(X.sum(1, keepdims=True), 1)


def balanced_pick(idx, lab, n, rng):
    """n indices out of `idx` with (near-)equal numbers per cluster label, water-filling when a cluster is smaller than its share."""
    groups = {c: rng.permutation(idx[lab[idx] == c]) for c in np.unique(lab[idx])}
    quota = {c: 0 for c in groups}
    left = n
    open_ = set(groups)
    while left > 0 and open_:
        share = max(left // len(open_), 1)
        for c in sorted(open_):
            take = min(share, len(groups[c]) - quota[c], left)
            quota[c] += take
            left -= take
            if quota[c] >= len(groups[c]):
                open_.discard(c)
            if left == 0:
                break
    return np.concatenate([groups[c][: quota[c]] for c in groups])


def badge_like(X, n, rng, batch=500, chunk=50_000):
    """Batched k-means++ seeding: each iteration draws `batch` points with probability proportional to squared distance to the chosen set."""
    N = len(X)
    sq = (X * X).sum(1)
    d2 = sq.copy()  # distance to the origin (the ensemble mean) before anything is chosen
    chosen = np.zeros(N, bool)
    out = []
    while len(out) < n:
        p = np.where(chosen, 0.0, d2).astype(np.float64)
        p /= p.sum()
        b = min(batch, n - len(out))
        pick = rng.choice(N, b, replace=False, p=p)
        chosen[pick] = True
        out.extend(pick.tolist())
        Xb = X[pick]
        sb = (Xb * Xb).sum(1)
        for lo in range(0, N, chunk):
            blk = X[lo : lo + chunk]
            d = sq[lo : lo + chunk, None] + sb[None, :] - 2 * blk @ Xb.T
            d2[lo : lo + chunk] = np.minimum(d2[lo : lo + chunk], np.maximum(d.min(1), 0))
        d2[chosen] = 0
    return np.array(out)


def gc_frac(seqs):
    return np.array([(q.count("G") + q.count("C")) / max(len(q), 1) for q in seqs])


def stratified_pick(strata, n, weights, rng):
    """n draws spread over strata proportionally to weights (water-filling when a stratum is too small), random within each stratum."""
    ids = np.unique(strata)
    size = np.array([(strata == i).sum() for i in ids])
    w = np.asarray(weights, float)
    quota = np.floor(n * w / w.sum()).astype(int)
    quota = np.minimum(quota, size)
    while quota.sum() < n:
        room = size - quota
        if room.sum() == 0:
            break
        j = int(np.argmax(np.where(room > 0, w / (quota + 1), -1)))
        quota[j] += 1
    return np.concatenate(
        [
            rng.choice(np.where(strata == i)[0], q, replace=False)
            for i, q in zip(ids, quota)
            if q > 0
        ]
    )


def criteria_arms(c, P):
    rng = np.random.default_rng(0)
    mu, sd = P.mean(0), P.std(0)
    sets = {}
    nb = 20
    e20 = np.quantile(mu, np.linspace(0, 1, nb + 1))
    b20 = np.clip(np.searchsorted(e20[1:-1], mu, side="right"), 0, nb - 1)
    resid = np.zeros_like(sd)
    for i in range(nb):
        m = b20 == i
        centre = np.median(sd[m])
        mad = np.median(np.abs(sd[m] - centre))
        scale = 1.4826 * mad if mad > 0 else sd[m].std()
        resid[m] = (sd[m] - centre) / scale if scale > 0 else 0.0
    sets["resid"] = np.argsort(-resid)[:N_ADD]
    edges = np.quantile(mu, np.linspace(0, 1, 6))
    b = np.clip(np.searchsorted(edges, mu, side="right") - 1, 0, 4)
    per = N_ADD // 5
    sets["binned"] = np.concatenate(
        [np.where(b == i)[0][np.argsort(-sd[b == i])[:per]] for i in range(5)]
    )
    log("k-mer features + k-means ...")
    from sklearn.cluster import MiniBatchKMeans

    X = kmer_features([str(s) for s in c["sequences"]])
    km = MiniBatchKMeans(N_CLUSTERS, random_state=0, batch_size=8192, n_init=1, max_iter=60).fit(X)
    lab = km.labels_
    allidx = np.arange(len(mu))
    sets["diverse"] = balanced_pick(allidx, lab, N_ADD, rng)
    top2 = np.argsort(-sd)[: 2 * N_ADD]
    sets["stdiv"] = balanced_pick(top2, lab, N_ADD, rng)
    log("badge-like k-means++ ...")
    sets["badge"] = badge_like((P - mu).T.astype(np.float32), N_ADD, rng)
    y = np.asarray(c["oracle_labels"], float)
    sets["oracle"] = np.argsort(-np.abs(y - mu))[:N_ADD]
    gcv = gc_frac([str(q) for q in c["sequences"]])
    gb = np.digitize(gcv, np.quantile(gcv, [1 / 3, 2 / 3]))
    ab = np.clip(np.searchsorted(edges[1:-1], mu, side="right"), 0, 4)
    s2 = ab * 3 + gb
    sets["strat2d"] = stratified_pick(s2, N_ADD, np.ones(15), rng)
    s3 = np.asarray(c["source"], int) * 15 + s2
    ids3 = np.unique(s3)
    sets["neyman"] = stratified_pick(
        s3, N_ADD, [(s3 == i).sum() * sd[s3 == i].mean() for i in ids3], rng
    )
    p = sd**2 / (sd**2).sum()
    sets["powersd"] = rng.choice(len(sd), N_ADD, replace=False, p=p)
    return sets


def main() -> None:
    ACQ.mkdir(exist_ok=True)
    base = np.load(REPS / "genomic_r1_d30000.npz", allow_pickle=True)
    bs, bl = list(base["sequences"][:30_000]), base["oracle_labels"][:30_000].astype(np.float32)
    for name, f in RESERVOIR_ARMS.items():
        dst = ACQ / f"res_{name}_d100000.npz"
        if dst.exists() or not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        assert str(z["oracle_id"]) == "oracle_v3"
        save(
            dst,
            bs + list(z["sequences"][:70_000]),
            np.concatenate([bl, z["oracle_labels"][:70_000].astype(np.float32)]),
            f"res_{name}",
        )
        log("wrote", dst.name)
    wanted = [
        "resid",
        "binned",
        "diverse",
        "stdiv",
        "badge",
        "oracle",
        "strat2d",
        "neyman",
        "powersd",
    ]
    todo = [a for a in wanted if not (ACQ / f"sel_{a}_d100000.npz").exists()]
    if not todo:
        log("criteria arms already built")
        return
    c = np.load(ACQ / "candidates_r1.npz", allow_pickle=True)
    P, used = load_scores()
    log(f"{len(used)} selection models x {P.shape[1]:,} candidates")
    assert len(used) >= 25 and P.shape[1] == len(c["sequences"])
    sets = criteria_arms(c, P)
    info, idxs = {"models": used, "arms": {}}, {}
    for a, sel in sets.items():
        if a not in todo:  # files of arms that exist may already be in use by training jobs
            continue
        sel = np.sort(np.unique(sel))
        assert len(sel) == N_ADD, (a, len(sel))
        seqs = bs + list(c["sequences"][sel])
        lab = np.concatenate([bl, c["oracle_labels"][sel]])
        save(ACQ / f"sel_{a}_d100000.npz", seqs, lab, f"acq_{a}")
        info["arms"][a] = {
            "label_mean": float(c["oracle_labels"][sel].mean()),
            "source_frac": (np.bincount(c["source"][sel], minlength=3) / len(sel)).tolist(),
            "n_unique": len(set(seqs)),
            "mean_sd": float(P.std(0)[sel].mean()),
        }
        idxs[a] = sel
        log(a, json.dumps(info["arms"][a]))
    z = np.load(ACQ / "selection_indices.npz")
    allsets = {**{k: z[k] for k in ("rand", "fullmean", "fullstd")}, **idxs}
    names = list(allsets)
    info["jaccard"] = {
        a: {
            b: len(np.intersect1d(allsets[a], allsets[b])) / len(np.union1d(allsets[a], allsets[b]))
            for b in names
        }
        for a in names
    }
    json.dump(info, open(ACQ / "selection_info2.json", "w"), indent=1)
    np.savez(ACQ / "selection_indices2.npz", **idxs)


if __name__ == "__main__":
    main()
