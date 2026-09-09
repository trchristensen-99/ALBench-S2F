"""Factsheet for the new/updated reservoir strategies: capacity and summary stats.

For each strategy: where its sequences come from, how many distinct sequences it can
produce, and the summary statistics that say what it actually generates (GC, distance
from source, duplicate rate, motif content). Sized at 20k per arm, which is enough
for stable statistics and cheap enough to run in one job.
"""

import logging
import sys
import time
import numpy as np

logging.disable(logging.INFO)
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")

N = 20_000
REPO = "/grid/wsbs/home_norepl/christen/ALBench-S2F"


def gc(seqs):
    return np.array([(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs])


def kmer_entropy(seqs, k=4, n=4000):
    from collections import Counter

    c = Counter()
    for s in seqs[:n]:
        for i in range(len(s) - k + 1):
            c[s[i : i + k]] += 1
    v = np.array(list(c.values()), dtype=float)
    v /= v.sum()
    return float(-(v * np.log2(v)).sum()), len(c)


def report(name, seqs, meta, extra="", t=None):
    g = gc(seqs)
    H, nk = kmer_entropy(seqs)
    dup = 1 - len(set(seqs)) / len(seqs)
    print(
        f"  {name:<38} n={len(seqs):>6,}  len={len(seqs[0]):>3}  "
        f"GC={g.mean():.3f}+-{g.std():.3f}  dup={dup:>6.2%}  "
        f"4mer H={H:.3f} ({nk}/256)" + (f"  [{t:.0f}s]" if t else "")
    )
    if extra:
        print(f"      {extra}")


print("=" * 104)
print("1. PHYLOGENETIC ZOONOMIA")
print("=" * 104)
from albench.reservoir.motif_planted_v2 import PhylogeneticZoonomiaSampler

z = np.load(f"{REPO}/data/zoonomia/per_position_rates.npz", allow_pickle=True)
rates = np.nan_to_num(np.asarray(z["subst_rate"], float))
print(f"  SOURCE: data/zoonomia/per_position_rates.npz")
print(f"    regions            {rates.shape[0]:,} cCRE x {rates.shape[1]}bp")
print(f"    species            {int(z['n_species'])} mammals")
print(
    f"    chroms excluded    {list(np.asarray(z['excluded_chroms']).astype(str))} (test) upstream"
)
cc = np.asarray(z["ccre_class"]).astype(str)
u, c = np.unique(cc, return_counts=True)
print(
    f"    cCRE classes       "
    + ", ".join(f"{a}={b:,}" for a, b in sorted(zip(u, c), key=lambda x: -x[1]))
)
print(
    f"    per-position rate  mean={rates.mean():.4f} median={np.median(rates):.4f} "
    f"p10={np.quantile(rates, 0.1):.4f} p90={np.quantile(rates, 0.9):.4f}"
)
print(
    f"    CAPACITY           {rates.shape[0]:,} distinct backgrounds x stochastic "
    f"mutation -> effectively unbounded"
)
print()
base = [str(s) for s in z["sequences"][:60_000]]
for mode, rate in (("flat", 0.02), ("per_position_matched", 0.02), ("per_position", None)):
    s = PhylogeneticZoonomiaSampler(seed=1, rate_mode=mode, mut_rate=rate)
    t0 = time.time()
    seqs, meta = s.generate(N, base_sequences=base) if mode == "flat" else s.generate(N)
    el = time.time() - t0
    ex = (
        f"mut/200bp mean={meta.n_mutations.mean():.1f} sd={meta.n_mutations.std():.2f} "
        f"({meta.n_mutations.mean() / 200:.2%}/pos)"
    )
    if "mean_phylop" in meta:
        ex += (
            f"; corr(phyloP,n_mut)={np.corrcoef(meta.mean_phylop, meta.n_mutations)[0, 1]:+.3f}"
            f"; regions used={meta.region_idx.nunique():,}"
        )
    report(f"rate_mode={mode}", seqs, meta, ex, el)

print()
print("=" * 104)
print("2. ENCODE ACCESSIBILITY")
print("=" * 104)
from albench.reservoir.encode_accessibility import EncodeAccessibilitySampler, PARTITIONS

for part in ("shared_open_both", "k562_only", "hepg2_only"):
    for mw in (0, 150):
        s = EncodeAccessibilitySampler(seed=1, partition=part, min_peak_width=mw)
        pk = s._load_peaks()
        t0 = time.time()
        seqs, meta = s.generate(min(N, len(pk)))
        el = time.time() - t0
        w = meta.peak_width
        report(
            f"{part} min_width={mw}",
            seqs,
            meta,
            f"usable peaks={len(pk):,} (CAPACITY); peak width med={int(w.median())} "
            f"p10={int(w.quantile(0.1))}; chroms={meta.chrom.nunique()}",
            el,
        )

print()
print("=" * 104)
print("3. MOTIF PLANTED v2 (JASPAR vocabulary)")
print("=" * 104)
from albench.reservoir.motif_planted_v2 import MotifPlantedV2Sampler

for cm in ("representative", "sample_members"):
    for pm in ("pwm_sample", "consensus"):
        s = MotifPlantedV2Sampler(
            seed=1,
            motif_set="jaspar",
            vocab_cluster_at=0.90,
            vocab_trim_ic=0.5,
            vocab_max_len=12,
            vocab_size=80,
            cluster_mode=cm,
            plant_mode=pm,
        )
        t0 = time.time()
        seqs, meta = s.generate(N, task="k562")
        el = time.time() - t0
        n_inst = [len(set(v)) for v in s._inst.values()]
        planted = [x for row in meta.planted_motifs for x in row.split(",") if x != "none"]
        report(
            f"cluster_mode={cm} plant_mode={pm}",
            seqs,
            meta,
            f"|V|={meta.vocab_size.iloc[0]} entries; distinct strings/entry "
            f"med={int(np.median(n_inst))} max={max(n_inst)}; "
            f"planted/seq={meta.n_motifs_planted.mean():.2f}; "
            f"distinct entries used={len(set(planted))}",
            el,
        )

print()
print("=" * 104)
print("4. GC-MATCHED RANDOM")
print("=" * 104)
from albench.reservoir.gc_matched import GCMatchedSampler

pool = [
    str(s)
    for s in np.load(f"{REPO}/outputs/chr_split_cache/chr_train_ref_only.npz", allow_pickle=True)[
        "sequences"
    ][:50_000]
]
pg = gc(pool)
print(f"  SOURCE: chr_train genomic pool for the GC target distribution")
print(f"    pool GC mean={pg.mean():.3f} sd={pg.std():.3f}")
print(f"    CAPACITY: generated de novo -> unbounded")
from scipy.stats import ks_2samp

for nb in (2, 10, 50):
    s = GCMatchedSampler(seed=1, n_gc_bins=nb)
    t0 = time.time()
    seqs, meta = s.generate(N, pool)
    el = time.time() - t0
    report(
        f"n_gc_bins={nb}",
        seqs,
        meta,
        f"KS(actual GC vs pool GC)={ks_2samp(meta.actual_gc, pg).statistic:.4f}",
        el,
    )

print()
print("=" * 104)
print("5. MOTIF CLUSTERING (regulatory-architecture coverage)")
print("=" * 104)
from albench.reservoir.motif_clustering import MotifClusteringSampler

sub = pool[:20_000]
print(f"  SOURCE: selects FROM the genomic pool -> CAPACITY = pool size ({len(sub):,} here)")
for k in (10, 30):
    s = MotifClusteringSampler(seed=1, n_clusters=k)
    t0 = time.time()
    seqs, meta = s.generate(8000, sub)
    el = time.time() - t0
    cs = meta.cluster_id.value_counts()
    report(
        f"n_clusters={k}",
        seqs,
        meta,
        f"clusters used={meta.cluster_id.nunique()}; per-cluster n "
        f"min={cs.min()} max={cs.max()}; motifs/seq={meta.n_motifs.mean():.2f}",
        el,
    )
print("\nDONE")
