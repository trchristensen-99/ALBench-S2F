#!/usr/bin/env python
"""Build validation-set VARIANTS (V0..V3), oracle-labelled with oracle_v3, with a measured similarity guard.

Why: every experiment so far early-stops, weights ensembles and gives the LLM proposer feedback on ONE validation set,
real genomic chr19/21/X sequences. That set may reward genomic-like models. These variants let us test whether the
choice matters without adding a new validation set per strategy:

  V0  genomic (chr19/21/X)                      the current baseline, unchanged sequences
  V1  perturbed genomic                         PRM point mutagenesis + EvoAug (published settings) of the V0 sequences
  V2  diverse families                          random@GC, gc_matched, dinuc shuffle, motif-planted, PRM at 4 rates,
                                                EvoAug x2, zoonomia orthologs (val chromosomes), oracle-guided evolution
  V3  union                                     equal thirds of V0, V1, V2 with no parent/child pair inside it

VALIDATION-SIDE TWINS. The samplers read their genomic parents from a pool the caller passes (registry adapters) or from
RESERVOIR_BG_CACHE / ALBENCH_BG_CACHE (motif planting), exactly as scripts/build_matched_test_sets.py inverts the test
side. Here every parent pool is chr_val_ref_only.npz; zoonomia's chromosome exclusion is inverted to keep ONLY chr19/21/X.

SIMILARITY GUARD. Every non-genomic variant is filtered so that no sequence shares >= tau of its canonical 12-mers with any
sequence of any evaluation set or any training pool (see kmer_sim.py). tau and the repeat-mask cap come from `calibrate`,
which measures the max-similarity distribution of unrelated sequences against perturbed copies; nothing is hand-picked.

Reproduce (the operating point actually used: k=12, repeat-mask cap 300, tau 0.10; k, cap and tau were chosen by `calibrate`, see
outputs/val_variants/calibration_k{12,16,20,24}/calibration.json):
  sbatch --export=ALL,STAGE=index,ARGS="--k 12" scripts/val_variants/cpu.sbatch
  sbatch --export=ALL,STAGE=calibrate,ARGS="--k 12 --cap 300 --tau 0.10" scripts/val_variants/cpu.sbatch
  sbatch --export=ALL,STAGE=gen,ARGS="--k 12" scripts/val_variants/cpu.sbatch
  sbatch scripts/val_variants/gpu.sbatch                      # oracle-guided evolution + oracle_v3 labelling; waits for `gen`
  python scripts/val_variants/build_val_variants.py assemble --k 12 && python scripts/val_variants/build_val_variants.py stats --k 12

Stages (each resumable; heavy ones run under sbatch):
  index      build + save the 12-mer index of all reference sequences          (CPU, ~10 min)
  calibrate  measure unrelated-vs-copy similarity, choose tau                   (CPU)
  gen        generate + filter every family except oracle-guided evolution      (CPU)
  gpu        oracle-guided evolution candidates + label everything              (1 GPU)
  assemble   finish filtering, build + stamp V0..V3, diversity statistics       (CPU)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
OUT = REPO / "outputs/val_variants"
VAL_NPZ = REPO / "outputs/chr_split_cache/chr_val_ref_only.npz"
TRAIN_NPZ = REPO / "outputs/chr_split_cache/chr_train_ref_only.npz"
BATTERY = REPO / "data/k562/test_sets_ag_s2_chrsplit"
V3DIR = REPO / "outputs/eval_sets_v3/labelled"
REPS = REPO / "outputs/grid_pools/k562_reps"
SEQ_LEN = 200
SEED0 = 20261023
ORACLE_ID = "oracle_v3"
K = 16  # k-mer length of the similarity index; set from --k, chosen by measurement (see calibrate)
TEST_CHROMS = ("chr7", "chr13")
VAL_CHROMS = ("chr19", "chr21", "chrX")
ALL_CHROMS = tuple(f"chr{i}" for i in list(range(1, 23)) + ["X"])
# the 18 evaluation sets students are scored on (snv splits into ref/alt)
EVAL18 = [
    "genomic",
    "ood",
    "snv_ref",
    "snv_alt",
    "random_32k",
    "gc_sweep",
    "dinuc_shuffle",
    "sub_low",
    "sub_med",
    "sub_high",
    "ins_low",
    "ins_med",
    "ins_high",
    "del_low",
    "del_med",
    "del_high",
    "translocation",
    "inversion",
]
CAL_RATES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
TAU_GRID = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]
OVERSAMPLE = 1.25


# ----------------------------------------------------------------------------------------------- reference collection
def _seqs(z, key):
    return [str(s) for s in z[key]]


def load_refs():
    """name -> sequences, name -> kind. Kinds: eval18 | eval_extra | eval_v3 | train."""
    named, kind = {}, {}
    for f in sorted(
        BATTERY.glob("*_oracle.npz")
    ):  # *.premix backups end in .premix, so they are not matched
        nm = f.name[: -len("_oracle.npz")]
        z = np.load(f, allow_pickle=True)
        if "ref_sequences" in z.files:
            named["snv_ref"], named["snv_alt"] = (
                _seqs(z, "ref_sequences"),
                _seqs(z, "alt_sequences"),
            )
        elif "sequences" in z.files:
            named[nm] = _seqs(z, "sequences")
    for nm in named:
        kind[nm] = "eval18" if nm in EVAL18 else "eval_extra"
    # eval_sets_v3: ONLY the designed set. v3 genomic / snv_* / random10k are the older pre-split sets and overlap the VALIDATION
    # chromosomes (measured: 9% of the V0 sequences share >=50% of their 12-mers with v3 genomic / v3 snv_ref), so they are not references.
    z = np.load(V3DIR / "ood.npz", allow_pickle=True)
    named["v3_ood"], kind["v3_ood"] = _seqs(z, "sequences"), "eval_v3"
    for f in sorted(REPS.glob("*_r1_d30000.npz")):
        named[f"pool_{f.name.split('_r1_')[0]}"] = _seqs(np.load(f, allow_pickle=True), "sequences")
        kind[f"pool_{f.name.split('_r1_')[0]}"] = "train"
    named["chr_train_genomic"] = _seqs(np.load(TRAIN_NPZ, allow_pickle=True), "sequences")
    kind["chr_train_genomic"] = "train"
    missing = [e for e in EVAL18 if e not in named]
    if missing:
        raise SystemExit(f"evaluation sets missing from {BATTERY}: {missing}")
    return named, kind


def _rc(s: str) -> str:
    return s[::-1].translate(str.maketrans("ACGTacgt", "TGCAtgca"))


def canon_str(s: str) -> str:
    r = _rc(s)
    return s if s <= r else r


def cmd_index(args):
    from kmer_sim import KmerIndex

    t0 = time.time()
    named, kind = load_refs()
    names = list(named)
    assert len(names) <= 64, "source bitmask is 64 bits"
    uniq: dict[str, int] = {}
    lens = {}
    for i, n in enumerate(names):
        bit = 1 << i
        for s in named[n]:
            lens[len(s)] = lens.get(len(s), 0) + 1
            s = s[:SEQ_LEN]
            uniq[s] = uniq.get(s, 0) | bit
    print(
        f"{len(names)} sources, {sum(len(v) for v in named.values()):,} rows, {len(uniq):,} unique sequences; lengths {dict(sorted(lens.items())[:6])}",
        flush=True,
    )
    idx = KmerIndex(K, SEQ_LEN)
    idx.add(list(uniq), np.array(list(uniq.values()), dtype=np.uint64))
    idx.finalize()
    idx.save(OUT / f"index{K}")
    meta = {
        "names": names,
        "kind": kind,
        "n_rows": {n: len(v) for n, v in named.items()},
        "n_unique_total": len(uniq),
        "length_histogram": {str(k): v for k, v in sorted(lens.items())},
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "refs": {
            "battery": str(BATTERY.relative_to(REPO)),
            "v3": str(V3DIR.relative_to(REPO)) + "/ood.npz only",
            "pools": "k562_reps/*_r1_d30000.npz",
            "chr_train": str(TRAIN_NPZ.relative_to(REPO)),
        },
        "build_seconds": round(time.time() - t0, 1),
    }
    (OUT / f"index{K}" / "meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"index built in {time.time() - t0:.0f}s: {idx.n_ref:,} refs, {len(idx.ids):,} postings",
        flush=True,
    )


def load_index():
    from kmer_sim import KmerIndex

    idx = KmerIndex.load(OUT / f"index{K}")
    meta = json.loads((OUT / f"index{K}/meta.json").read_text())
    names = meta["names"]
    kmask = {}
    for i, n in enumerate(names):
        kmask[meta["kind"][n]] = kmask.get(meta["kind"][n], 0) | (1 << i)
    return idx, meta, {k: np.uint64(v) for k, v in kmask.items()}


def describe_hits(idx, meta, kmask, best):
    """kind of the best-matching reference (and its first source name) per query."""
    src = np.where(best >= 0, idx.src[np.maximum(best, 0)], np.uint64(0))
    kinds = np.full(len(best), "none", dtype=object)
    for k in ("eval18", "eval_extra", "eval_v3", "train"):  # priority order for the label
        hit = (src & kmask.get(k, np.uint64(0))) != 0
        kinds[hit & (kinds == "none")] = k
    first = np.full(len(best), "", dtype=object)
    for i, n in reversed(list(enumerate(meta["names"]))):
        first[(src >> np.uint64(i)) & np.uint64(1) == 1] = n
    return kinds, first


# ---------------------------------------------------------------------------------------------------- generation
def val_sequences():
    z = np.load(VAL_NPZ, allow_pickle=True)
    return [str(s) for s in z["sequences"]], np.asarray(z["oracle_labels"], np.float32)


def use_val_backgrounds():
    """Point every sampler that reads a genomic background pool at the VALIDATION chromosomes (the twin trick of
    scripts/build_matched_test_sets.py, inverted to the validation side)."""
    os.environ["RESERVOIR_BG_CACHE"] = str(VAL_NPZ)
    os.environ["ALBENCH_BG_CACHE"] = str(VAL_NPZ)


def gen_registry(strategy, n, seed, pool, **params):
    from albench.core.registry import Context, get

    spec = get(strategy)
    ctx = Context(task="k562", pool_sequences=pool)
    seqs, meta = spec.generate(n, ctx, seed=seed, **params)
    return [str(s) for s in seqs], meta


def gen_zoonomia(n, seed, max_species_per_window=3):
    from albench.reservoir.zoonomia_orthologs import ZoonomiaOrthologSampler

    held = tuple(
        c for c in ALL_CHROMS if c not in VAL_CHROMS
    )  # inverted exclusion: keep ONLY chr19/21/X
    s = ZoonomiaOrthologSampler(
        seed=seed,
        exclude_chroms=held,
        max_species_per_window=max_species_per_window,
        min_called_frac=0.95,
    )
    seqs, meta = s.generate(n, task="k562")
    bad = set(meta["chrom"]) - set(VAL_CHROMS)
    assert not bad, f"zoonomia windows outside validation chromosomes: {bad}"
    return [str(x) for x in seqs], meta


# family table: key -> (variant, strategy, params, quota). subfamily labels are derived per row.
V1_FAMILIES = [
    ("prm_0.05", "mutagenesis", {"mutation_rate": 0.05}, 14725),
    ("evoaug_paper2023", "evoaug_paper2023", {}, 14724),
]
V2_FAMILIES = [
    (
        "random_gc",
        "random_gc_sweep",
        {"gc_levels": "0.30,0.40,0.50,0.60,0.70"},
        4150,
    ),  # 830 per level
    ("gc_matched", "gc_matched", {}, 1500),
    ("dinuc_shuffle", "dinuc_shuffle", {}, 3500),
    ("motif_ct_enriched", "motif_ct_enriched", {}, 2400),
    ("motif_shared_core", "motif_shared_core", {}, 2000),
    ("motif_syntax_core", "motif_syntax_core", {}, 1600),
    ("prm_0.01", "mutagenesis", {"mutation_rate": 0.01}, 1300),
    ("prm_0.05", "mutagenesis", {"mutation_rate": 0.05}, 1300),
    ("prm_0.10", "mutagenesis", {"mutation_rate": 0.10}, 1300),
    ("prm_0.20", "mutagenesis", {"mutation_rate": 0.20}, 1300),
    ("evoaug_paper2023", "evoaug_paper2023", {}, 1800),
    ("evoaug_ours_default", "evoaug_ours_default", {}, 1800),
    ("zoonomia_orthologs_valchr", "zoonomia", {}, 3000),
    (
        "ise_oracle_maximize",
        "ise_maximize",
        {
            "n_evolution_rounds": 5,
            "population_size": 24,
            "mutation_rate": 0.03,
            "elite_fraction": 0.25,
        },
        2499,
    ),
]


def family_table():
    rows = [("V1",) + f for f in V1_FAMILIES] + [("V2",) + f for f in V2_FAMILIES]
    return rows


def gen_family(strategy, params, n, seed, val_seqs):
    """Return (sequences, per-row arrays). parent_idx indexes the V0 sequences (or -1 if the family has no genomic parent)."""
    if strategy == "zoonomia":
        seqs, meta = gen_zoonomia(n, seed)
        return seqs, {
            "sub": np.array([f"{c}" for c in meta["chrom"]], dtype=object),
            "parent": np.full(len(seqs), -1, np.int64),
        }
    seqs, meta = gen_registry(strategy, n, seed, val_seqs, **params)
    parent = (
        np.asarray(meta["parent_idx"], np.int64)
        if "parent_idx" in meta.columns
        else np.full(len(seqs), -1, np.int64)
    )
    sub = (
        np.array([f"gc{g:.2f}" for g in meta["target_gc"]], dtype=object)
        if (strategy == "random_gc_sweep" and "target_gc" in meta.columns)
        else np.full(len(seqs), "", dtype=object)
    )
    return seqs, {"sub": sub, "parent": parent}


def sim_filter(idx, meta, kmask, seqs, tau, cap):
    shared, nk, best, nmask = idx.query(seqs, cap=cap)
    sim = shared / np.maximum(nk, 1)
    kinds, first = describe_hits(idx, meta, kmask, best)
    return sim, kinds, first, nk, nmask


def v0_root(val_idx, seqs):
    """For rows with no recorded parent: the V0 sequence they are derived from, if any (>=0.25 shared 12-mers)."""
    shared, nk, best, _ = val_idx.query(seqs, cap=10**9)
    s = shared / np.maximum(nk, 1)
    return np.where(s >= 0.25, best, -1), s.astype(np.float32)


def cmd_gen(args):
    from kmer_sim import KmerIndex, canonical_kmers, encode  # noqa: F401

    cal = json.loads((OUT / f"calibration_k{K}/calibration.json").read_text())
    tau = args.tau if args.tau is not None else cal["chosen"]["tau"]
    cap = args.cap if args.cap is not None else cal["chosen"]["cap"]
    print(f"similarity guard: tau={tau} cap={cap}", flush=True)
    idx, meta, kmask = load_index()
    use_val_backgrounds()
    val_seqs, _ = val_sequences()
    OUT.joinpath("filtered").mkdir(exist_ok=True)
    seen: set[str] = set()  # exact (strand-collapsed) duplicates across everything we emit
    for s in val_seqs:
        seen.add(canon_str(s))
    rows = {
        k: []
        for k in ("seq", "variant", "family", "sub", "parent", "sim", "kind", "src", "attempt")
    }
    report = {}
    for vi, (variant, fam, strategy, params, quota) in enumerate(
        [(r[0], r[1], r[2], r[3], r[4]) for r in family_table()]
    ):
        if strategy == "ise_maximize":
            continue  # needs the oracle: generated in the GPU stage
        quota = max(20, int(quota * args.scale)) if args.scale != 1.0 else quota
        t0 = time.time()
        got, rep = (
            [],
            {
                "quota": quota,
                "generated": 0,
                "dup": 0,
                "sim_removed": 0,
                "removed_by_kind": {},
                "kept": 0,
            },
        )
        for attempt in range(4):
            need = quota - len(got)
            if need <= 0:
                break
            n_gen = int(np.ceil(OVERSAMPLE * need)) + 50
            seed = SEED0 + 1000 * vi + 100 * attempt
            seqs, extra = gen_family(strategy, params, n_gen, seed, val_seqs)
            rep["generated"] += len(seqs)
            keep = []
            for i, s in enumerate(seqs):
                c = canon_str(s)
                if len(s) != SEQ_LEN or c in seen:
                    rep["dup"] += 1
                    continue
                keep.append(i)
                seen.add(c)
            sim, kinds, first, nk, nmask = sim_filter(
                idx, meta, kmask, [seqs[i] for i in keep], tau, cap
            )
            for j, i in enumerate(keep):
                if sim[j] >= tau:
                    rep["sim_removed"] += 1
                    rep["removed_by_kind"][kinds[j]] = rep["removed_by_kind"].get(kinds[j], 0) + 1
                    continue
                got.append(
                    (
                        seqs[i],
                        extra["sub"][i],
                        int(extra["parent"][i]),
                        float(sim[j]),
                        kinds[j],
                        first[j],
                        attempt,
                    )
                )
        # trim to the quota; for random_gc keep an equal number per GC level
        if fam == "random_gc":
            levels = sorted({g[1] for g in got})
            per = quota // len(levels)
            take, cnt = [], {lv: 0 for lv in levels}
            for i, g in enumerate(got):
                if cnt[g[1]] < per:
                    cnt[g[1]] += 1
                    take.append(i)
            rest = [i for i in range(len(got)) if i not in set(take)][: quota - len(take)]
            got = [got[i] for i in sorted(take + rest)]
        else:
            got = got[:quota]
        rep["kept"] = len(got)
        report[f"{variant}/{fam}"] = rep
        for g in got:
            rows["seq"].append(g[0])
            rows["variant"].append(variant)
            rows["family"].append(fam)
            rows["sub"].append(g[1])
            rows["parent"].append(g[2])
            rows["sim"].append(g[3])
            rows["kind"].append(g[4])
            rows["src"].append(g[5])
            rows["attempt"].append(g[6])
        print(
            f"  {variant}/{fam:<28s} quota={quota:>6,} generated={rep['generated']:>7,} dup={rep['dup']:>5,} sim_removed={rep['sim_removed']:>5,} "
            f"kept={len(got):>6,}  {time.time() - t0:.0f}s",
            flush=True,
        )
    # parents for rows that do not record one (motif planting keeps part of a V0 background): detect via a V0-only index

    vidx = KmerIndex(K, SEQ_LEN)
    vidx.add(val_seqs, np.ones(len(val_seqs), np.uint64))
    vidx.finalize()
    parent = np.array(rows["parent"], np.int64)
    need = np.flatnonzero(parent < 0)
    root, rsim = v0_root(vidx, [rows["seq"][i] for i in need])
    parent_detected = parent.copy()
    parent_detected[need] = root
    rsim_all = np.zeros(len(parent), np.float32)
    rsim_all[need] = rsim
    np.savez(
        OUT / f"filtered/to_label{args.tag}.npz",
        sequences=np.array(rows["seq"], dtype=object),
        variant=np.array(rows["variant"]),
        family=np.array(rows["family"]),
        sub=np.array(rows["sub"], dtype=object),
        parent_idx=parent_detected,
        v0_sim=rsim_all,
        max_sim=np.array(rows["sim"], np.float32),
        hit_kind=np.array(rows["kind"], dtype=object),
        hit_source=np.array(rows["src"], dtype=object),
        attempt=np.array(rows["attempt"], np.int16),
        tau=np.float32(tau),
        cap=np.int64(cap),
    )
    (OUT / f"filtered/gen_report{args.tag}.json").write_text(
        json.dumps({"tau": tau, "cap": cap, "families": report}, indent=1)
    )
    print(
        f"wrote {len(rows['seq']):,} candidate sequences -> {OUT / f'filtered/to_label{args.tag}.npz'}",
        flush=True,
    )


# ------------------------------------------------------------------------------------------------------ calibration
def _quantiles(x):
    x = np.asarray(x, float)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q90": float(np.quantile(x, 0.90)),
        "q99": float(np.quantile(x, 0.99)),
        "q999": float(np.quantile(x, 0.999)),
        "max": float(x.max()),
    }


def cmd_calibrate(args):
    t0 = time.time()
    idx, meta, kmask = load_index()
    use_val_backgrounds()
    rng = np.random.default_rng(SEED0)
    val_seqs, _ = val_sequences()
    test = _seqs(np.load(BATTERY / "genomic_oracle.npz", allow_pickle=True), "sequences")
    parents = [
        test[i] for i in rng.choice(len(test), 3000, replace=False)
    ]  # chr7/13 sequences, all inside the index
    Q = {}
    from kmer_sim import SENT, canonical_kmers, encode

    def kmer_refcounts(seqs):
        can = canonical_kmers(encode(seqs, SEQ_LEN), idx.k)
        pos = np.minimum(np.searchsorted(idx.ukeys, can), len(idx.ukeys) - 1)
        return idx.counts[pos][(can != SENT) & (idx.ukeys[pos] == can)]

    refcount_q = {
        n: {
            str(q): float(np.quantile(kmer_refcounts(sq), q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        }
        for n, sq in (
            ("chr7_13_test_genomic", parents),
            ("chr19_21_X_val_genomic_V0", val_seqs[:3000]),
        )
    }
    print(
        "number of reference sequences containing a k-mer, quantiles:",
        json.dumps(refcount_q),
        flush=True,
    )
    Q["unrelated/iid_random"], _ = gen_registry("random", 3000, SEED0 + 1, None)
    Q["unrelated/genomic_val_V0"] = val_seqs
    for r in CAL_RATES:
        Q[f"copy/prm_{r:.2f}"], _ = gen_registry(
            "mutagenesis", 3000, SEED0 + 2, parents, mutation_rate=r
        )
    Q["copy/evoaug_paper2023"], _ = gen_registry("evoaug_paper2023", 3000, SEED0 + 3, parents)
    Q["copy/evoaug_ours_default"], _ = gen_registry("evoaug_ours_default", 3000, SEED0 + 4, parents)
    Q["copy/reverse_complement"] = [_rc(s) for s in parents]
    Q["control/dinuc_shuffle_of_parents"], _ = gen_registry(
        "dinuc_shuffle", 3000, SEED0 + 5, parents
    )
    caps = sorted(set([10, 30, 100, 300, 1000, args.cap]))
    res, raw = (
        {
            "k": idx.k,
            "n_reference_sequences": int(idx.n_ref),
            "kmer_refcount_quantiles": refcount_q,
        },
        {},
    )
    for name, seqs in Q.items():
        seqs = [s[:SEQ_LEN] for s in seqs]
        for cap in caps + [10**9]:
            sub = seqs
            if cap >= 1000 and len(seqs) > 3000:
                sub = [
                    seqs[i] for i in np.random.default_rng(1).choice(len(seqs), 3000, replace=False)
                ]
            if cap == 10**9 and len(seqs) > 1500:
                sub = [
                    seqs[i] for i in np.random.default_rng(2).choice(len(seqs), 1500, replace=False)
                ]
            tq = time.time()
            shared, nk, best, nmask = idx.query(sub, cap=cap)
            sim = shared / np.maximum(nk, 1)
            kinds, first = describe_hits(idx, meta, kmask, best)
            key = f"{name}|cap={'inf' if cap == 10**9 else cap}"
            q = _quantiles(sim)
            q["shared_median"] = float(np.median(shared))
            q["shared_q99"] = float(np.quantile(shared, 0.99))
            q["frac_masked_kmers"] = float((nmask / np.maximum(nk + nmask, 1)).mean())
            q["frac_ge"] = {str(t): float((sim >= t).mean()) for t in TAU_GRID}
            q["hit_kind_at_q99"] = {
                k: int(((kinds == k) & (sim >= np.quantile(sim, 0.99))).sum())
                for k in ("eval18", "eval_extra", "eval_v3", "train")
            }
            res[key] = q
            raw[key] = sim.astype(np.float32)
            print(
                f"  {key:<52s} n={len(sub):>6,} median={q['median']:.3f} q99={q['q99']:.3f} max={q['max']:.3f}  ({time.time() - tq:.0f}s)",
                flush=True,
            )
    # choose tau from the CHANCE floor, per cap: smallest tau (step 0.01, never below tau_min) that no more than `fpr_chance` of
    # iid-random sequences and of dinucleotide-shuffled copies reach. Real homology in genomic sequence (repeat families with few
    # copies, segmental duplications) is NOT noise: it is reported as the fraction of V0 that the guard would remove, not used to
    # raise tau, because a V1/V2 sequence derived from such a parent is exactly what the guard is meant to catch.
    chosen = None
    for cap in caps:
        cs = f"cap={cap}"
        rnd, shuf, g = (
            raw[f"unrelated/iid_random|{cs}"],
            raw[f"control/dinuc_shuffle_of_parents|{cs}"],
            raw[f"unrelated/genomic_val_V0|{cs}"],
        )
        for tau in [round(x, 2) for x in np.arange(args.tau_min, 0.51, 0.01)]:
            if (rnd >= tau).mean() <= args.fpr_chance and (shuf >= tau).mean() <= args.fpr_chance:
                res[f"_rule|{cs}"] = {
                    "tau": tau,
                    "fpr_random": float((rnd >= tau).mean()),
                    "fpr_dinuc_shuffle": float((shuf >= tau).mean()),
                    "frac_V0_genomic_removed": float((g >= tau).mean()),
                    "detect": {
                        k.split("|")[0]: float((raw[k] >= tau).mean())
                        for k in raw
                        if k.endswith(cs) and k.startswith(("copy/", "control/"))
                    },
                }
                if cap == args.cap and chosen is None:
                    chosen = {"cap": cap, **res[f"_rule|{cs}"]}
                break
    if (
        args.tau is not None
    ):  # explicit operating point, validated against the measured chance floor
        cs = f"cap={args.cap}"
        rnd, shuf, g = (
            raw[f"unrelated/iid_random|{cs}"],
            raw[f"control/dinuc_shuffle_of_parents|{cs}"],
            raw[f"unrelated/genomic_val_V0|{cs}"],
        )
        floor = float(max(rnd.max(), shuf.max()))
        assert args.tau > floor, (
            f"tau={args.tau} is not above the chance floor {floor:.3f} (max over iid-random and dinucleotide-shuffled queries)"
        )
        chosen = {
            "cap": args.cap,
            "tau": args.tau,
            "chance_floor": floor,
            "fpr_random": float((rnd >= args.tau).mean()),
            "fpr_dinuc_shuffle": float((shuf >= args.tau).mean()),
            "frac_V0_genomic_removed": float((g >= args.tau).mean()),
            "detect": {
                k.split("|")[0]: float((raw[k] >= args.tau).mean())
                for k in raw
                if k.endswith(cs) and k.startswith(("copy/", "control/"))
            },
        }
        shared_v0, nk_v0, best_v0, _ = idx.query(val_seqs, cap=args.cap)
        sim_v0 = shared_v0 / np.maximum(nk_v0, 1)
        kinds_v0, _ = describe_hits(idx, meta, kmask, best_v0)
        chosen["V0_removed_by_hit_kind"] = {
            k: int(((sim_v0 >= args.tau) & (kinds_v0 == k)).sum())
            for k in ("eval18", "eval_extra", "eval_v3", "train")
        }
        chosen["V0_n_removed"] = int((sim_v0 >= args.tau).sum())
        chosen["V0_n"] = len(val_seqs)
    if chosen is None:
        raise SystemExit(
            "no tau on the grid meets the chance-level false-positive target; raise --tau_min range or relax --fpr_chance"
        )
    res["chosen"] = chosen
    res["rule"] = (
        f"tau = smallest value >= {args.tau_min} (step 0.01) with P(iid random >= tau) <= {args.fpr_chance} and "
        f"P(dinuc-shuffled >= tau) <= {args.fpr_chance}; repeat-mask cap={args.cap}; k={idx.k}"
    )
    d = OUT / f"calibration_k{K}"
    d.mkdir(exist_ok=True, parents=True)
    (d / "calibration.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(
        d / "sims.npz", **{k.replace("|", "__").replace("/", "_"): v for k, v in raw.items()}
    )
    print(f"calibration done in {time.time() - t0:.0f}s; chosen = {chosen}", flush=True)


# ------------------------------------------------------------------------------------------------ GPU stage + assembly
def cmd_gpu(args):
    """Oracle-guided evolution candidates, then label every candidate (and the V0 sequences) with the oracle_v3 ensemble."""
    t0 = time.time()
    use_val_backgrounds()
    from experiments.exp1_1_scaling import _load_oracle
    from scripts.generate_labeled_pools import _label_sequences

    print(f"AG_S2_ORACLE_DIR={os.environ.get('AG_S2_ORACLE_DIR')}", flush=True)
    oracle = _load_oracle("k562", oracle_type="ag_s2")
    val_seqs, _ = val_sequences()
    n_ise = int(np.ceil(1.2 * [r for r in V2_FAMILIES if r[0] == "ise_oracle_maximize"][0][3])) + 50
    from albench.core.registry import Context, get

    spec = get("ise_maximize")
    ise_params = [r for r in V2_FAMILIES if r[0] == "ise_oracle_maximize"][0][2]
    ctx = Context(task="k562", pool_sequences=val_seqs, oracle=oracle)
    tq = time.time()
    ise, imeta = spec.generate(n_ise, ctx, seed=SEED0 + 777, **ise_params)
    print(f"ISE: {len(ise):,} evolved sequences in {time.time() - tq:.0f}s", flush=True)
    (OUT / "raw").mkdir(exist_ok=True, parents=True)
    ise_lab_fit = np.asarray(imeta["final_fitness"], np.float32)
    np.savez(
        OUT / "raw/ise_candidates.npz",
        sequences=np.array(ise, dtype=object),
        parent_idx=np.asarray(imeta["parent_idx"], np.int64),
        final_fitness=ise_lab_fit,
        params=json.dumps(ise_params),
    )

    def label(fname, seqs):
        dst = OUT / f"raw/{fname}.npz"
        if dst.exists():
            print(f"skip {fname}: exists", flush=True)
            return
        tq = time.time()
        lab = np.asarray(_label_sequences(oracle, seqs), np.float32).ravel()
        if not np.isfinite(lab).all():
            raise SystemExit(
                f"{fname}: {(~np.isfinite(lab)).sum()} non-finite labels; refusing to write"
            )
        tmp = str(dst) + ".tmp.npz"
        np.savez_compressed(
            tmp,
            sequences=np.array(seqs, dtype=object),
            oracle_labels=lab,
            oracle_id=ORACLE_ID,
            oracle_dir=os.environ.get("AG_S2_ORACLE_DIR", ""),
        )
        os.replace(tmp, dst)
        print(
            f"labelled {fname}: n={len(seqs):,} mean={lab.mean():.3f} sd={lab.std():.3f} ({time.time() - tq:.0f}s, {len(seqs) / (time.time() - tq):.0f} seq/s)",
            flush=True,
        )

    label("labelled_ise", [str(s) for s in ise])
    label(
        "V0_relabel_oracle_v3", val_seqs
    )  # same name/keys scripts/label_pool.py writes, so either path can produce it
    f = OUT / "filtered/to_label.npz"
    tw = time.time()
    while (
        not f.exists() and time.time() - tw < args.wait_min * 60
    ):  # the CPU stage may still be queued; hold the slot rather than queue for a GPU twice
        time.sleep(30)
    if f.exists():
        label("labelled_to_label", _seqs(np.load(f, allow_pickle=True), "sequences"))
    else:
        print(
            f"filtered/to_label.npz did not appear within {args.wait_min} min; run `gpu` again after `gen`",
            flush=True,
        )
    print(f"gpu stage done in {time.time() - t0:.0f}s", flush=True)


def _sha(seqs, labels):
    h = hashlib.sha1()
    for s in seqs:
        h.update(str(s).encode())
    h.update(np.asarray(labels, np.float32).tobytes())
    return h.hexdigest()[:16]


def cmd_assemble(args):
    from albench.composition import cpg_obs_exp, gc_content_batch

    t0 = time.time()
    cal = json.loads((OUT / f"calibration_k{K}/calibration.json").read_text())
    tau, cap = cal["chosen"]["tau"], cal["chosen"]["cap"]
    idx, meta, kmask = load_index()
    val_seqs, val_stored = val_sequences()
    v0 = np.load(OUT / "raw/V0_relabel_oracle_v3.npz", allow_pickle=True)
    assert [str(s) for s in v0["sequences"]] == val_seqs
    v0_new = np.asarray(v0["oracle_labels"], np.float32)
    cl = np.load(OUT / "raw/labelled_to_label.npz", allow_pickle=True)
    cand = np.load(OUT / "filtered/to_label.npz", allow_pickle=True)
    assert [str(s) for s in cl["sequences"]] == [str(s) for s in cand["sequences"]]
    seqs = [str(s) for s in cand["sequences"]]
    lab = np.asarray(cl["oracle_labels"], np.float32)
    fam, var, sub = (
        np.array(cand["family"]),
        np.array(cand["variant"]),
        np.array(cand["sub"], dtype=object),
    )
    sub[fam != "random_gc"] = np.where(
        fam[fam != "random_gc"] == "zoonomia_orthologs_valchr", sub[fam != "random_gc"], ""
    )  # per-row GC targets of gc_matched are not sub-families
    parent = np.asarray(cand["parent_idx"], np.int64)
    maxsim = np.asarray(cand["max_sim"], np.float32)
    kind, hsrc = (
        np.array(cand["hit_kind"], dtype=object),
        np.array(cand["hit_source"], dtype=object),
    )
    # ---- oracle-guided evolution: finish the guard here (generated on the GPU, after the index existed)
    ise = np.load(OUT / "raw/ise_candidates.npz", allow_pickle=True)
    il = np.load(OUT / "raw/labelled_ise.npz", allow_pickle=True)
    iseqs = [str(s) for s in ise["sequences"]]
    sim, kinds, first, nk, nmask = sim_filter(idx, meta, kmask, iseqs, tau, cap)
    seen = {canon_str(s) for s in val_seqs} | {canon_str(s) for s in seqs}
    ok = []
    for j, s in enumerate(iseqs):
        c = canon_str(s)
        if sim[j] >= tau or c in seen:
            continue
        seen.add(c)
        ok.append(j)
    quota = [r for r in V2_FAMILIES if r[0] == "ise_oracle_maximize"][0][3]
    print(
        f"ISE: {len(iseqs)} candidates, {len(ok)} pass the guard (tau={tau}); keeping {min(quota, len(ok))}",
        flush=True,
    )
    ok = ok[:quota]
    ilab = np.asarray(il["oracle_labels"], np.float32)
    seqs += [iseqs[j] for j in ok]
    lab = np.concatenate([lab, ilab[ok]])
    fam = np.concatenate([fam, np.array(["ise_oracle_maximize"] * len(ok))])
    var = np.concatenate([var, np.array(["V2"] * len(ok))])
    sub = np.concatenate([sub, np.array([""] * len(ok), dtype=object)])
    parent = np.concatenate([parent, np.asarray(ise["parent_idx"], np.int64)[ok]])
    maxsim = np.concatenate([maxsim, sim[ok].astype(np.float32)])
    kind = np.concatenate([kind, kinds[ok]])
    hsrc = np.concatenate([hsrc, first[ok]])
    # ---- stamp + write V0, V1, V2, V3
    rng = np.random.default_rng(SEED0 + 5)
    prov_common = {
        "oracle_id": ORACLE_ID,
        "oracle_dir": str(v0["oracle_dir"]),
        "similarity": {
            "k": K,
            "tau": tau,
            "cap": cap,
            "calibration": f"outputs/val_variants/calibration_k{K}/calibration.json",
        },
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_val_pool": "outputs/chr_split_cache/chr_val_ref_only.npz (chr19/21/X, ref alleles)",
    }
    try:
        prov_common["git_head"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    def write(name, s, y, family, subf, par, ms, desc):
        s = [str(x) for x in s]
        assert len(s) == len(y) == len(family) and np.isfinite(y).all()
        prov = dict(
            prov_common,
            variant=name,
            description=desc,
            n=len(s),
            sha1_16=_sha(s, y),
            families={k: int((np.array(family) == k).sum()) for k in sorted(set(family))},
        )
        np.savez_compressed(
            OUT / f"val_{name}.npz",
            sequences=np.array(s, dtype=object),
            oracle_labels=np.asarray(y, np.float32),
            oracle_id=ORACLE_ID,
            family=np.array(family),
            subfamily=np.array(subf, dtype=object),
            parent_v0_idx=np.asarray(par, np.int64),
            max_sim=np.asarray(ms, np.float32),
            provenance=json.dumps(prov),
        )
        print(
            f"wrote val_{name}.npz n={len(s):,} mean={np.mean(y):.3f} sd={np.std(y):.3f}",
            flush=True,
        )
        return prov

    prov = {}
    n0 = len(val_seqs)
    # V0: the validation set the pipeline uses today. Stored labels are what the default code path loads; the oracle_v3 relabel
    # is written alongside so the label-space question can be answered by measurement.
    prov["V0"] = write(
        "V0",
        val_seqs,
        v0_new,
        ["genomic_val"] * n0,
        [""] * n0,
        np.arange(n0),
        np.zeros(n0),
        "chr19/21/X genomic validation sequences, labels RE-SCORED with oracle_v3",
    )
    np.savez_compressed(
        OUT / "val_V0_stored_labels.npz",
        sequences=np.array(val_seqs, dtype=object),
        oracle_labels=val_stored,
        oracle_id="UNSTAMPED_stored_in_chr_val_ref_only",
    )
    m1, m2 = var == "V1", var == "V2"
    prov["V1"] = write(
        "V1",
        np.array(seqs, dtype=object)[m1],
        lab[m1],
        fam[m1],
        sub[m1],
        parent[m1],
        maxsim[m1],
        "PRM (rate 0.05) + EvoAug paper2023 perturbations of the V0 sequences",
    )
    prov["V2"] = write(
        "V2",
        np.array(seqs, dtype=object)[m2],
        lab[m2],
        fam[m2],
        sub[m2],
        parent[m2],
        maxsim[m2],
        "diverse strategy families built on chr19/21/X backgrounds",
    )
    # V3: equal thirds, no V0 parent used twice and no parent/child pair inside the set
    third = n0 // 3
    i1 = np.flatnonzero(m1)
    i2 = np.flatnonzero(m2)
    used: set[int] = set()
    pick1 = []
    for i in rng.permutation(i1):
        p = int(parent[i])
        if p >= 0 and p in used:
            continue
        pick1.append(i)
        if p >= 0:
            used.add(p)
        if len(pick1) == third:
            break
    pick2 = []
    for i in rng.permutation(i2):
        p = int(parent[i])
        if p >= 0 and p in used:
            continue
        pick2.append(i)
        if p >= 0:
            used.add(p)
        if len(pick2) == third:
            break
    free0 = np.array([j for j in rng.permutation(n0) if j not in used])[
        : n0 - len(pick1) - len(pick2)
    ]
    s3 = [seqs[i] for i in pick1] + [seqs[i] for i in pick2] + [val_seqs[j] for j in free0]
    y3 = np.concatenate([lab[pick1], lab[pick2], v0_new[free0]])
    f3 = list(fam[pick1]) + list(fam[pick2]) + ["genomic_val"] * len(free0)
    sub3 = list(sub[pick1]) + list(sub[pick2]) + [""] * len(free0)
    par3 = np.concatenate([parent[pick1], parent[pick2], free0])
    ms3 = np.concatenate([maxsim[pick1], maxsim[pick2], np.zeros(len(free0))])
    prov["V3"] = write(
        "V3",
        s3,
        y3,
        f3,
        sub3,
        par3,
        ms3,
        "equal thirds of V1, V2 and V0; no V0 parent appears twice or alongside its own derivative",
    )
    (OUT / "VARIANTS.json").write_text(json.dumps(prov, indent=1))
    print(f"assembled in {time.time() - t0:.0f}s", flush=True)


def cmd_stats(args):
    """Diversity statistics per variant and, for context, per evaluation set."""
    from scipy.stats import wasserstein_distance

    from albench.composition import cpg_obs_exp, gc_content_batch

    out, comp = {}, {}

    def summ(name, seqs, y, fam=None):
        gc = gc_content_batch(seqs)
        cpg = np.array([cpg_obs_exp(s) for s in seqs])

        def q(a):
            names = ("q05", "q25", "q50", "q75", "q95")
            return {
                k: float(v) for k, v in zip(names, np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95]))
            }

        d = {
            "n": len(seqs),
            "gc": dict(mean=float(gc.mean()), sd=float(gc.std()), **q(gc)),
            "cpg_oe": dict(mean=float(cpg.mean()), sd=float(cpg.std()), **q(cpg)),
            "label": dict(mean=float(np.mean(y)), sd=float(np.std(y)), **q(y)),
            "frac_gc_outside_0.25_0.72": float(((gc < 0.25) | (gc > 0.72)).mean()),
            "n_unique": len(set(seqs)),
        }
        if fam is not None:
            d["family_counts"] = {k: int((np.array(fam) == k).sum()) for k in sorted(set(fam))}
        comp[name] = (gc, np.asarray(y, float))
        return d

    named, kind = load_refs()
    canon_sets = {n: {canon_str(x[:SEQ_LEN]) for x in sq} for n, sq in named.items()}
    train_chr = canon_sets["chr_train_genomic"]
    test_chr = canon_sets["genomic"]
    for v in ("V0", "V1", "V2", "V3"):
        z = np.load(OUT / f"val_{v}.npz", allow_pickle=True)
        sq = [str(s) for s in z["sequences"]]
        out[v] = summ(v, sq, z["oracle_labels"], z["family"])
        cs = [canon_str(x) for x in sq]
        # leakage checks, measured: exact (either strand) overlap with every reference collection, incl. the train chromosomes and the chr7/13 test genomic set
        out[v]["exact_overlap_with"] = {
            n: int(sum(c in st for c in cs))
            for n, st in canon_sets.items()
            if any(c in st for c in cs)
        }
        out[v]["exact_overlap_chr_train"] = int(sum(c in train_chr for c in cs))
        out[v]["exact_overlap_chr7_13_test_genomic"] = int(sum(c in test_chr for c in cs))
        ms = np.asarray(z["max_sim"], float)
        out[v]["max_sim_to_any_reference"] = (
            {
                "q50": float(np.quantile(ms, 0.5)),
                "q99": float(np.quantile(ms, 0.99)),
                "max": float(ms.max()),
            }
            if v != "V0"
            else None
        )
        par = np.asarray(z["parent_v0_idx"])
        out[v]["frac_with_V0_parent"] = float((par >= 0).mean())
        out[v]["n_distinct_V0_parents"] = int(len(set(par[par >= 0].tolist())))
    evalsets = {}
    for e in EVAL18:
        f = BATTERY / f"{e.replace('snv_ref', 'snv').replace('snv_alt', 'snv')}_oracle.npz"
        z = np.load(f, allow_pickle=True)
        if e.startswith("snv"):
            sq, yy = (
                _seqs(z, "ref_sequences" if e == "snv_ref" else "alt_sequences"),
                np.asarray(z["ref_mean" if e == "snv_ref" else "alt_mean"], float),
            )
        else:
            sq, yy = _seqs(z, "sequences"), np.asarray(z["oracle_mean"], float)
        evalsets[e] = summ(f"eval/{e}", sq, yy)
    out["eval_sets"] = evalsets
    # how much of each evaluation set's GC / label distribution does each variant cover? (1-D Wasserstein, smaller = closer)
    out["wasserstein_gc"] = {
        v: {e: float(wasserstein_distance(comp[v][0], comp[f"eval/{e}"][0])) for e in EVAL18}
        for v in ("V0", "V1", "V2", "V3")
    }
    out["wasserstein_label"] = {
        v: {e: float(wasserstein_distance(comp[v][1], comp[f"eval/{e}"][1])) for e in EVAL18}
        for v in ("V0", "V1", "V2", "V3")
    }
    (OUT / "diversity_stats.json").write_text(json.dumps(out, indent=1))
    for v in ("V0", "V1", "V2", "V3"):
        d = out[v]
        print(
            f"{v}: n={d['n']:,} GC {d['gc']['mean']:.3f}+-{d['gc']['sd']:.3f} [q05 {d['gc']['q05']:.2f} q95 {d['gc']['q95']:.2f}] CpG o/e {d['cpg_oe']['mean']:.2f}+-{d['cpg_oe']['sd']:.2f} "
            f"label {d['label']['mean']:.3f}+-{d['label']['sd']:.3f} outside-GC-support {d['frac_gc_outside_0.25_0.72']:.3f} unique {d['n_unique']:,}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("index", cmd_index),
        ("calibrate", cmd_calibrate),
        ("gen", cmd_gen),
        ("gpu", cmd_gpu),
        ("assemble", cmd_assemble),
        ("stats", cmd_stats),
    ):
        p = sp.add_parser(name)
        p.set_defaults(fn=fn)
        if name in ("calibrate",):
            p.add_argument("--cap", type=int, default=50)
            p.add_argument("--fpr_chance", type=float, default=0.0005)
            p.add_argument("--tau_min", type=float, default=0.05)
            p.add_argument(
                "--tau",
                type=float,
                default=None,
                help="explicit operating point (must exceed the measured chance floor); default: chance-floor rule",
            )
        if name == "gpu":
            p.add_argument(
                "--wait_min",
                type=float,
                default=90.0,
                help="how long to hold the GPU waiting for the CPU `gen` stage output",
            )
        if name == "gen":
            p.add_argument("--tau", type=float, default=None)
            p.add_argument("--cap", type=int, default=None)
            p.add_argument(
                "--scale", type=float, default=1.0, help="quota multiplier for smoke tests"
            )
            p.add_argument("--tag", default="", help="suffix of the output files (smoke tests)")
    for p in sp.choices.values():
        p.add_argument("--k", type=int, default=16, help="k-mer length of the similarity index")
    args = ap.parse_args()
    global K
    K = args.k
    OUT.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
