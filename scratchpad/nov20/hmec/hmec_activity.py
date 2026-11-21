"""Turn the HMEC read counts into activity, and check the result against published numbers.

We counted these reads ourselves from fastq, because ENCODE released no quantification for
this experiment. That means nothing downstream has been checked by anyone, so the activity
is computed and validated in one place rather than trusted.

Activity is log2( RNA CPM / DNA CPM ) per replicate, with a pseudocount, then combined across
the five replicates; the standard error is the spread across replicates, which is the honest
one here because we have no UMIs and therefore cannot separate PCR duplicates from
independent molecules. Counts, not barcodes, so the SE is a replicate SE and not a
within-library one.

Four checks, each able to fail independently:

1. REPLICATE AGREEMENT. Five RNA replicates of the same library should correlate highly with
   each other. If they do not, the counting is wrong or the libraries are not what we think.
2. DNA UNIFORMITY vs the published ``DNA_mean``. The plasmid pool is the same designed
   library the Tewhey experiments used, so our DNA counts should track their published DNA
   abundance for the same oligo IDs. This tests the SEQUENCE->OLIGO assignment directly,
   independent of any biology.
3. CROSS-CELL AGREEMENT. HMEC activity should correlate with published activity in the seven
   immortalised lines on shared oligos -- strongly enough to show we are measuring promoter
   activity, not so strongly that we are accidentally reproducing another cell's file.
4. ALLELIC SANITY. ref and alt oligos of the same variant should agree closely, since most
   variants do nothing; a near-zero correlation would mean the allele assignment is noise.

    python3 -I hmec_activity.py --counts outputs/hmec/counts \
        --design data/external/tewhey_elements/ENCSR091QEV__ENCFF770UJN.fa \
                 data/external/tewhey_elements/ENCSR935CNL__ENCFF090JTW.fa \
        --quant data/external/tewhey_encode --out outputs/hmec/hmec_activity.npz
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

PSEUDO = 1.0
MIN_DNA = 20  # an oligo with almost no plasmid reads has an undefined ratio, not a low one


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rna_f = sorted(glob.glob(os.path.join(a.counts, "rna_rep*.npz")))
    dna_f = sorted(glob.glob(os.path.join(a.counts, "dna_rep*.npz")))
    if not rna_f or not dna_f:
        raise SystemExit(f"no count files under {a.counts}")

    z0 = np.load(rna_f[0], allow_pickle=True)
    seqs = np.array([str(s) for s in z0["sequences"]], dtype=object)
    names = np.array([str(s) for s in z0["names"]], dtype=object)
    n = len(seqs)

    def stack(files):
        out = np.zeros((len(files), n), np.float64)
        for i, p in enumerate(files):
            z = np.load(p, allow_pickle=True)
            if len(z["counts"]) != n or not np.array_equal(
                np.array([str(s) for s in z["sequences"]], dtype=object), seqs
            ):
                raise SystemExit(f"{p} indexes a different design set")
            out[i] = z["counts"]
        return out

    rna, dna = stack(rna_f), stack(dna_f)
    print(
        f"{n:,} designs | RNA {rna.shape[0]} reps, {rna.sum():,.0f} reads"
        f" | DNA {dna.shape[0]} reps, {dna.sum():,.0f} reads\n",
        flush=True,
    )

    cpm_r = rna / rna.sum(1, keepdims=True) * 1e6
    cpm_d = dna / dna.sum(1, keepdims=True) * 1e6
    dna_mean = cpm_d.mean(0)
    usable = (dna.sum(0) >= MIN_DNA) & (rna.sum(0) > 0)

    # One log-ratio per RNA replicate against the pooled plasmid, then combine.
    lfc_rep = np.log2((cpm_r + PSEUDO) / (dna_mean[None, :] + PSEUDO))
    lfc = lfc_rep.mean(0)
    lfc_se = lfc_rep.std(0, ddof=1) / np.sqrt(lfc_rep.shape[0])
    lfc[~usable] = np.nan
    lfc_se[~usable] = np.nan
    print(
        f"{int(usable.sum()):,} of {n:,} designs usable (>= {MIN_DNA} plasmid reads and >0 RNA)\n",
        flush=True,
    )

    report: dict = {"n_designs": int(n), "n_usable": int(usable.sum())}

    print("=== 1. replicate agreement (RNA log-ratio, pairwise r)")
    rs = []
    for i in range(lfc_rep.shape[0]):
        for j in range(i + 1, lfc_rep.shape[0]):
            r = pearson(lfc_rep[i][usable], lfc_rep[j][usable])
            rs.append(r)
    print(f"   {len(rs)} pairs: min {min(rs):.4f}  median {np.median(rs):.4f}  max {max(rs):.4f}")
    report["replicate_r"] = {"min": round(min(rs), 4), "median": round(float(np.median(rs)), 4)}

    print("\n=== 2. our plasmid (DNA) abundance vs published DNA_mean, same oligo IDs")
    id_of = {}
    for i, nm in enumerate(names):
        for one in str(nm).split(";"):
            id_of.setdefault(one, i)
    pub_dna: dict[int, list[float]] = collections.defaultdict(list)
    pub_lfc: dict[str, dict[int, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for p in sorted(glob.glob(os.path.join(a.quant, "*.tsv"))):
        cell = os.path.basename(p).split("__")[0]
        cols = pd.read_csv(p, sep="\t", nrows=0).columns
        use = [c for c in ("ID", "DNA_mean", "log2FoldChange") if c in cols]
        if "ID" not in use:
            continue
        d = pd.read_csv(p, sep="\t", usecols=use, low_memory=False)
        ids = d["ID"].astype(str).to_numpy()
        if "DNA_mean" in d:
            v = d["DNA_mean"].to_numpy(np.float64)
            for k, oid in enumerate(ids):
                i = id_of.get(oid)
                if i is not None and np.isfinite(v[k]):
                    pub_dna[i].append(v[k])
        if "log2FoldChange" in d:
            v = d["log2FoldChange"].to_numpy(np.float64)
            for k, oid in enumerate(ids):
                i = id_of.get(oid)
                if i is not None and np.isfinite(v[k]):
                    pub_lfc[cell][i].append(v[k])
    if pub_dna:
        idx = np.array(sorted(pub_dna))
        theirs = np.array([np.mean(pub_dna[i]) for i in idx])
        ours = dna_mean[idx]
        m = usable[idx] & (theirs > 0)
        r = pearson(np.log2(ours[m] + 1), np.log2(theirs[m] + 1))
        print(f"   {int(m.sum()):,} oligos matched by ID; r(log DNA) = {r:.4f}")
        report["dna_vs_published_r"] = round(r, 4)
    else:
        print("   no published DNA_mean matched by ID")

    print("\n=== 3. HMEC activity vs published activity, same oligo IDs")
    print("   %-10s %9s %9s" % ("cell", "n", "r"))
    cross = {}
    for cell, d in sorted(pub_lfc.items()):
        idx = np.array(sorted(d))
        theirs = np.array([np.mean(d[i]) for i in idx])
        m = np.isfinite(lfc[idx]) & np.isfinite(theirs)
        if m.sum() < 200:
            continue
        r = pearson(lfc[idx][m], theirs[m])
        cross[cell] = round(r, 4)
        print("   %-10s %9d %9.4f" % (cell, int(m.sum()), r))
    report["vs_published_activity"] = cross

    print("\n=== 4. allelic sanity: ref vs alt of the same variant")
    by_var: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for i, nm in enumerate(names):
        one = str(nm).split(";")[0]
        parts = one.split("_")
        if len(parts) >= 2 and parts[1] in ("ref", "alt"):
            by_var["_".join([parts[0]] + parts[2:])][parts[1]] = i
    pairs = [(v["ref"], v["alt"]) for v in by_var.values() if "ref" in v and "alt" in v]
    if pairs:
        ri = np.array([p[0] for p in pairs])
        ai = np.array([p[1] for p in pairs])
        m = np.isfinite(lfc[ri]) & np.isfinite(lfc[ai])
        r = pearson(lfc[ri][m], lfc[ai][m])
        d = lfc[ai][m] - lfc[ri][m]
        print(
            f"   {int(m.sum()):,} ref/alt pairs; r = {r:.4f}; "
            f"alt-ref mean {d.mean():+.4f}, sd {d.std():.4f}"
        )
        report["ref_alt_r"] = round(r, 4)
        report["ref_alt_delta_sd"] = round(float(d.std()), 4)

    var = float(np.nanvar(lfc))
    noise = float(np.nanmean(np.square(lfc_se)))
    ceil = float(np.sqrt(max(0.0, 1 - noise / var))) if var > 0 else float("nan")
    print(
        f"\nHMEC activity: sd {np.sqrt(var):.3f}, mean SE {np.sqrt(noise):.3f}, "
        f"measurement ceiling {ceil:.4f}"
    )
    report["activity_sd"] = round(float(np.sqrt(var)), 4)
    report["mean_se"] = round(float(np.sqrt(noise)), 4)
    report["ceiling"] = round(ceil, 4)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(
        a.out,
        sequences=seqs,
        names=names,
        log2FC=lfc.astype(np.float32),
        log2FC_se=lfc_se.astype(np.float32),
        rna_counts=rna.astype(np.int64),
        dna_counts=dna.astype(np.int64),
        usable=usable,
        cell=np.array("HMEC", dtype=object),
    )
    with open(a.out.replace(".npz", "_qc.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
