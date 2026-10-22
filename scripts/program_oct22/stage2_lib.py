"""Shared helpers for the Oct-22 experiment program (independent-pool replicates, architecture-source test, mixtures, unseen reservoirs)."""
import glob, json, os
import numpy as np
R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
GP = f"{R}/outputs/grid_pools"
REPS = f"{GP}/k562_reps"
RAW30 = f"{GP}/k562_reps_raw30k"
UNSEEN = ["motif_shared_core", "motif_syntax_core", "zoonomia_orthologs", "mutagenesis", "gc_matched", "dinuc_shuffle"]
SEED = lambda r: 42 + r  # replicate r uses generation seed 42 + r (r1..r3 = 43..45, as built earlier)


def save_pool(path, seqs, labels, oracle_id, strategy, seed):
    tmp = str(path).replace(".npz", f".tmp{os.getpid()}.npz")
    np.savez(tmp, sequences=np.asarray(seqs, dtype=object), oracle_labels=np.asarray(labels, np.float32), oracle_id=oracle_id, strategy=strategy, seed=np.int64(seed))
    os.replace(tmp, path)


def make_genomic_reps(rmax=15):
    """Genomic replicates r4..rmax: seed-(42+r) permutations of the full labelled CRE set, prefixes at 10k/30k/100k."""
    gz = np.load(f"{GP}/k562_labelled/genomic_full/genomic.npz", allow_pickle=True)
    gs, gl = gz["sequences"], gz["oracle_labels"].astype(np.float32)
    for r in range(4, rmax + 1):
        if all(os.path.exists(f"{REPS}/genomic_r{r}_d{D}.npz") for D in (10000, 30000, 100000)):
            continue
        perm = np.random.default_rng(SEED(r)).permutation(len(gs))[:100000]
        for D in (10000, 30000, 100000):
            save_pool(f"{REPS}/genomic_r{r}_d{D}.npz", gs[perm[:D]], gl[perm[:D]], "oracle_v3", "genomic", SEED(r))


def slice30k():
    """Turn raw 30k pools built by build_pool.py into the standard replicate file names."""
    from albench.pools import PoolSpec
    n = 0
    jobs = [(st, r) for st in ("random", "motif_ct_enriched") for r in range(4, 16)] + [(st, r) for st in UNSEEN for r in (1, 2, 3)]
    for st, r in jobs:
        dst = f"{REPS}/{st}_r{r}_d30000.npz"
        if os.path.exists(dst):
            continue
        src = str(PoolSpec(strategy=st, params={}, size=30000, generation_seed=SEED(r), task="k562").path(RAW30))
        if not os.path.exists(src):
            continue
        z = np.load(src, allow_pickle=True)
        if len(z["sequences"]) < 30000 or str(z["oracle_id"]) != "oracle_v3" or not np.isfinite(z["oracle_labels"]).all():
            print("BAD raw pool", src); continue
        save_pool(dst, list(z["sequences"])[:30000], z["oracle_labels"][:30000], "oracle_v3", st, SEED(r)); n += 1
    return n
