"""Per-reservoir activity and k-mer characterisation for every arm.

Peter asked why some reservoirs work and others do not. Two descriptors separate
them: the ORACLE ACTIVITY distribution (does the arm place sequences where the
assay is informative, or pile them at the inactive floor?) and the K-MER
composition (is the arm producing genome-like sequence or synthetic sequence that
happens to be high-activity?).

Everything is computed against `genomic` as the reference arm, because the test
sets that matter are real genomic sequence. Reported per arm:
  activity: mean/sd/quantiles, fraction above the genomic 90th percentile
  k-mer   : 4-mer Shannon entropy, JS divergence of the 4-mer spectrum vs genomic
  overlap : 1 - total-variation distance between activity histograms vs genomic
"""
from __future__ import annotations
import glob, json, os
import numpy as np

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
POOLS = f"{ROOT}/outputs/grid_pools/k562_cumulative"
K = 4
NUC = {c: i for i, c in enumerate("ACGT")}


def kmer_spectrum(seqs, k=K, cap=4000):
    counts = np.zeros(4 ** k, dtype=np.float64)
    for s in seqs[:cap]:
        s = str(s).upper()
        idx = [NUC.get(c, -1) for c in s]
        for i in range(len(idx) - k + 1):
            w = idx[i : i + k]
            if -1 in w:
                continue
            v = 0
            for d in w:
                v = v * 4 + d
            counts[v] += 1
    tot = counts.sum()
    return counts / tot if tot else counts


def js(p, q, eps=1e-12):
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load(name):
    f = f"{POOLS}/{name}_d30000.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    if "sequences" not in z.files:
        return None
    lab = None
    for k in ("oracle_labels", "labels", "y"):
        if k in z.files:
            lab = np.asarray(z[k], dtype=np.float64).ravel(); break
    return z["sequences"], lab


names = sorted({os.path.basename(f).replace("_d30000.npz", "")
                for f in glob.glob(f"{POOLS}/*_d30000.npz")})
ref = load("genomic")
if ref is None:
    raise SystemExit("genomic reference pool missing")
ref_seq, ref_lab = ref
ref_spec = kmer_spectrum(ref_seq)
ref_hi = np.quantile(ref_lab, 0.90)
bins = np.linspace(np.min(ref_lab) - 1, np.max(ref_lab) + 1, 60)
ref_hist, _ = np.histogram(ref_lab, bins=bins, density=True)

rows = []
for n in names:
    d = load(n)
    if d is None:
        continue
    seq, lab = d
    spec = kmer_spectrum(seq)
    ent = float(-np.sum(spec[spec > 0] * np.log2(spec[spec > 0])))
    r = {"arm": n, "n": len(seq), "kmer_entropy": ent,
         "kmer_js_vs_genomic": js(spec, ref_spec)}
    if lab is not None and len(lab) == len(seq):
        h, _ = np.histogram(lab, bins=bins, density=True)
        tv = 0.5 * np.sum(np.abs(h - ref_hist)) * (bins[1] - bins[0])
        r.update(act_mean=float(lab.mean()), act_sd=float(lab.std()),
                 act_p10=float(np.quantile(lab, .1)), act_p90=float(np.quantile(lab, .9)),
                 frac_above_genomic_p90=float((lab > ref_hi).mean()),
                 activity_overlap_vs_genomic=float(1 - min(tv, 1)))
    rows.append(r)

rows.sort(key=lambda r: -r.get("frac_above_genomic_p90", -1))
print("\n  %-22s %7s %8s %8s %8s %9s %9s" % (
    "arm", "act_mu", "act_sd", ">gen_p90", "act_ovl", "4mer_ent", "4mer_JS"))
for r in rows:
    print("  %-22s %7s %8s %8s %8s %9.4f %9.4f" % (
        r["arm"],
        f"{r['act_mean']:.3f}" if "act_mean" in r else "—",
        f"{r['act_sd']:.3f}" if "act_sd" in r else "—",
        f"{r['frac_above_genomic_p90']:.3f}" if "frac_above_genomic_p90" in r else "—",
        f"{r['activity_overlap_vs_genomic']:.3f}" if "activity_overlap_vs_genomic" in r else "—",
        r["kmer_entropy"], r["kmer_js_vs_genomic"]))
print(f"\n  genomic reference: p90={ref_hi:.3f}  4mer entropy={float(-np.sum(ref_spec[ref_spec>0]*np.log2(ref_spec[ref_spec>0]))):.4f}")
os.makedirs(f"{ROOT}/outputs/analysis", exist_ok=True)
json.dump(rows, open(f"{ROOT}/outputs/analysis/strategy_characterization.json", "w"), indent=2)
print(f"  wrote outputs/analysis/strategy_characterization.json ({len(rows)} arms)")
