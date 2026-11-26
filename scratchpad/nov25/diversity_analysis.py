"""What uncertainty actually takes, in sequence terms, as the candidate pool grows.

fig14/fig17 answered this in predicted-activity terms and that is not enough: a rule that
ranked by activity alone would produce the same picture, and "designed sequences lack
diversity" is a claim about sequence content, not about a predicted number. This measures
the content directly, for three groups at every pool size:

    selected        the 30,000 uncertainty took
    not selected    everything else in the same candidate pool
    whole pool      the two together, as the reference both are read against

THE 30k POOL IS THE CONTROL AND IT IS NOT OPTIONAL. At a candidate pool of 30,000 the rule
selects 30,000, so "selected" and "whole pool" are the same set of sequences and every
metric here must come out identical. Anything that differs there is a bug in this script,
not a finding, and the check is asserted rather than eyeballed.

METRICS, and why each one is here:

  motifs per sequence       how much grammar is present at all
  distinct motifs per seq   whether the hits are varied or one motif repeated
  effective motif count     exp(Shannon entropy) over the corpus-wide motif identity
                            distribution -- a Hill number, so it reads as "this corpus uses
                            N motifs equally often". This is the diversity metric: a corpus
                            that leans on a handful of strong motifs scores low even when
                            its raw motif count is high.
  6-mer entropy             the same idea without a motif vocabulary, so a drop cannot be
                            blamed on the vocabulary's coverage
  nearest-neighbour redundancy
                            share of sequences with a >=0.9 cosine 6-mer neighbour inside
                            their own group; this is what "lacks diversity" means if it
                            means anything concrete
  GC and CpG                composition confounds everything in this assay, so they are
                            reported rather than assumed constant

Scanning is a matmul. Sliding a PWM over one-hot sequence is a dot product per position, so
stacking the windows into one matrix turns the whole vocabulary into a few GEMMs and the
full CIS-BP set becomes seconds on a GPU instead of hours of python.

    python scratchpad/nov25/diversity_analysis.py --out diversity.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ALBENCH_REPO", Path(__file__).resolve().parents[2]))
NEST = REPO / "outputs/grid_pools/k562_nested"
POOLS = REPO / "outputs/acq52/pools"
SIZES = (30_000, 60_000, 100_000, 300_000, 1_000_000)
# poolA selects from motif_shared_core ONLY -- it is a pool-SIZE sweep, not a mixture
# sweep. Stated here because the figure it feeds has been read as a mixture more than once.
SOURCE = "motif_shared_core"
IDX = {c: i for i, c in enumerate("ACGT")}


def onehot(seqs: list[str], length: int) -> np.ndarray:
    """(N, 4, L) float32. Anything that is not ACGT becomes all-zero, which scores 0.

    Vectorised over the whole block. The per-character Python loop this replaces was the
    scan's real cost -- it ran once per motif WIDTH as well as once per chunk, so a
    vocabulary spanning fifteen widths encoded every sequence fifteen times.
    """
    buf = "".join(s[:length].ljust(length, "N") for s in seqs).encode("ascii", "replace")
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(seqs), length)
    out = np.zeros((len(seqs), 4, length), dtype=np.float32)
    for i, c in enumerate(b"ACGT"):
        out[:, i, :] = arr == c
    return out


def pval_threshold(lo: np.ndarray, p_target: float, binw: float = 0.01) -> float:
    """Score a motif must reach for a background p-value of `p_target`, computed exactly.

    A fixed "85% of maximum" cut was tried first and is useless here: with a 969-motif
    vocabulary it returned 391 hits and 178 distinct motifs per 200 bp, i.e. essentially
    every motif hitting every sequence, so the diversity numbers it produced were noise
    around a saturated scan.

    Under a uniform background each position contributes one of four log-odds values with
    probability 1/4, so the score distribution is the convolution of w four-point
    distributions and can be built exactly by DP over discretised scores. That makes the
    threshold a property of the motif rather than a knob, which matters because motifs here
    differ in width and information content and one shared cut would favour the long ones.
    """
    w = lo.shape[0]
    steps = np.rint(lo / binw).astype(np.int64)  # (w, 4)
    offset = int(-steps.min(axis=1).sum())  # shift so the lowest total is index 0
    size = int(steps.max(axis=1).sum()) + offset + 1
    cur = np.zeros(size)
    cur[offset] = 1.0  # the empty prefix scores 0, which sits at index `offset`
    for i in range(w):
        nxt = np.zeros(size)
        for s in steps[i]:
            sh = int(s)
            if sh >= 0:
                nxt[sh:] += 0.25 * cur[: size - sh]
            else:
                nxt[: size + sh] += 0.25 * cur[-sh:]
        cur = nxt
    tail = np.cumsum(cur[::-1])[::-1]  # P(S >= this bin)
    idx = int(np.argmax(tail <= p_target))
    if tail[idx] > p_target:  # never reached: demand the maximum score
        idx = size - 1
    return float((idx - offset) * binw)


def scan_counts(seqs: list[str], motifs, device: str, p_target: float = 3e-6, chunk: int = 4096):
    """(hits per sequence per motif) as an (N, M) int16 count matrix.

    A hit is a window whose log-odds score has background p-value <= `p_target` on either
    strand. At 3e-6 the whole vocabulary is expected to fire about once per sequence by
    chance (969 motifs x 2 strands x ~190 positions), so a count of five is signal rather
    than a threshold artefact.
    """
    import torch

    L = max(len(s) for s in seqs)
    by_width: dict[int, list[int]] = {}
    for mi, m in enumerate(motifs):
        by_width.setdefault(m.pwm.shape[0], []).append(mi)

    # Build every width's weight matrix and threshold ONCE, before touching a sequence.
    # The p-value DP is per motif and does not depend on the data, so recomputing it inside
    # the chunk loop was pure waste.
    prepared = {}
    for w, idxs in sorted(by_width.items()):
        if w > L:
            continue
        mats, thresh = [], []
        for mi in idxs:
            p = np.asarray(motifs[mi].pwm, dtype=np.float32)  # (w, 4)
            lo = np.log2((p + 1e-3) / 0.25)
            for mat in (lo, lo[::-1, ::-1]):  # forward and reverse complement
                mats.append(mat.T.reshape(-1))  # (4*w,)
            # Same threshold for both strands: the reverse complement is the same motif
            # and its score distribution under a uniform background is identical.
            t = pval_threshold(lo, p_target)
            thresh += [t, t]
        prepared[w] = (
            idxs,
            torch.tensor(np.stack(mats, axis=1), device=device),  # (4w, 2M_w)
            torch.tensor(np.asarray(thresh, dtype=np.float32), device=device),
        )

    counts = np.zeros((len(seqs), len(motifs)), dtype=np.int16)
    # Chunks outside, widths inside: one encoding of each block now serves every width,
    # where the previous nesting re-encoded the whole group once per width.
    for lo_i in range(0, len(seqs), chunk):
        block = seqs[lo_i : lo_i + chunk]
        x = torch.tensor(onehot(block, L), device=device)  # (B, 4, L)
        for w, (idxs, W, T) in prepared.items():
            # unfold gives every window; reshape so one GEMM scores every motif at every
            # position of every sequence in the block.
            win = x.unfold(2, w, 1)  # (B, 4, L-w+1, w)
            B, _, P, _ = win.shape
            win = win.permute(0, 2, 1, 3).reshape(B * P, 4 * w)
            sc = (win @ W).reshape(B, P, -1)  # (B, P, 2M_w)
            hit = (sc >= T).sum(dim=1)  # (B, 2M_w)
            hit = hit[:, 0::2] + hit[:, 1::2]  # fold the two strands together
            counts[lo_i : lo_i + len(block), idxs] = hit.to("cpu").numpy().astype(np.int16)
    return counts


def kmer_profile(seqs: list[str], k: int = 6) -> np.ndarray:
    """(N, 4**k) float32 row-normalised k-mer counts."""
    n4 = 4**k
    out = np.zeros((len(seqs), n4), dtype=np.float32)
    for i, s in enumerate(seqs):
        code, run = 0, 0
        for c in s:
            j = IDX.get(c)
            if j is None:
                code, run = 0, 0
                continue
            code = (code * 4 + j) % n4
            run += 1
            if run >= k:
                out[i, code] += 1
    tot = out.sum(axis=1, keepdims=True)
    np.divide(out, np.maximum(tot, 1), out=out)
    return out


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def redundancy(prof: np.ndarray, device: str, thresh: float = 0.9, cap: int = 4000) -> float:
    """Share of sequences with a >=thresh cosine 6-mer neighbour in their own group."""
    import torch

    if len(prof) > cap:
        sel = np.random.default_rng(0).choice(len(prof), cap, replace=False)
        prof = prof[sel]
    x = torch.tensor(prof, device=device)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
    hot = torch.zeros(len(x), dtype=torch.bool, device=device)
    for i in range(0, len(x), 512):
        s = x[i : i + 512] @ x.T
        s[torch.arange(len(s), device=device), torch.arange(i, i + len(s), device=device)] = -1
        hot[i : i + len(s)] = (s >= thresh).any(dim=1)
    return float(hot.float().mean())


def describe(seqs, labels, motifs, names, device, tag: str) -> dict:
    counts = scan_counts(seqs, motifs, device)
    per_seq = counts.sum(axis=1)
    distinct = (counts > 0).sum(axis=1)
    totals = counts.sum(axis=0).astype(np.float64)
    p = totals / max(1.0, totals.sum())
    prof = kmer_profile(seqs)
    pooled = prof.mean(axis=0)
    gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
    cg = np.array([s.count("CG") / max(1, len(s) - 1) for s in seqs])
    exp_cg = gc / 2 * gc / 2
    out = {
        "n": int(len(seqs)),
        "motifs_per_seq": round(float(per_seq.mean()), 3),
        "distinct_motifs_per_seq": round(float(distinct.mean()), 3),
        "effective_motif_count": round(float(math.exp(entropy(p))), 2),
        "motifs_used": int((totals > 0).sum()),
        "kmer6_entropy": round(entropy(pooled), 4),
        "redundancy_0.9": round(redundancy(prof, device), 4),
        "gc": round(float(gc.mean()), 4),
        "cpg_obs_exp": round(float(np.mean(cg / np.maximum(exp_cg, 1e-6))), 4),
        "activity_mean": round(float(np.mean(labels)), 4) if labels is not None else None,
        # Dynamic RANGE, not just level. A rule that shifts the mean up by taking only the
        # top of the pool teaches a model about one end of the activity axis; a rule that
        # widens the spread covers the axis. These are different corpora and the mean
        # cannot tell them apart, so the spread and the decile coverage are recorded too.
        "activity_sd": round(float(np.std(labels)), 4) if labels is not None else None,
        "activity_p10": round(float(np.quantile(labels, 0.10)), 4) if labels is not None else None,
        "activity_p90": round(float(np.quantile(labels, 0.90)), 4) if labels is not None else None,
        "activity_iqr": (
            round(float(np.quantile(labels, 0.75) - np.quantile(labels, 0.25)), 4)
            if labels is not None
            else None
        ),
        "top_motifs": [[names[i], round(float(p[i]), 5)] for i in np.argsort(-p)[:25] if p[i] > 0],
        "_motif_share": p.tolist(),
    }
    print(
        f"    {tag:<13} n={out['n']:>7,}  motifs/seq={out['motifs_per_seq']:>6.2f}  "
        f"distinct/seq={out['distinct_motifs_per_seq']:>5.2f}  "
        f"eff_motifs={out['effective_motif_count']:>7.2f}  "
        f"6mer_H={out['kmer6_entropy']:.3f}  redund={out['redundancy_0.9']:.3f}  "
        f"GC={out['gc']:.3f}",
        flush=True,
    )
    return out


def _rebuild_pool(srcs: list[str]) -> tuple[list[str], np.ndarray, bool]:
    """The candidate pool an arm chose from, from its `reservoir=N` source strings.

    select_corpus built these in memory and wrote only the selection, so the pool has to be
    reassembled from the sidecar. Pools are prefix-nested, so "random=75000" is the first
    75,000 rows of random's nested pool -- the same rows the arm was offered.
    """
    # "@/path/to/pool.npz" means the candidate pool is a file rather than a recipe, which
    # is how every run_al arm records it. Its seed prefix is dropped here because the
    # comparison is about what was CHOSEN from the candidates, and the prefix was never on
    # offer -- it was already labelled and trained on.
    if len(srcs) == 1 and str(srcs[0]).startswith("@"):
        z = np.load(str(srcs[0])[1:], allow_pickle=True)
        nb = int(z["n_base"]) if "n_base" in z.files else 0
        return (
            [str(x) for x in z["sequences"][nb:]],
            np.asarray(z["oracle_labels"][nb:], dtype=float),
            True,
        )
    seqs: list[str] = []
    labs: list[np.ndarray] = []
    for s in srcs:
        res, _, n = str(s).partition("=")
        want = int(n)
        cands = sorted(NEST.glob(f"{res}_d*.npz"), key=lambda p: p.stat().st_size)
        if not cands:
            return [], np.array([]), False
        z = np.load(cands[-1], allow_pickle=True)
        if len(z["sequences"]) < want:
            return [], np.array([]), False
        seqs += [str(x) for x in z["sequences"][:want]]
        labs.append(np.asarray(z["oracle_labels"][:want], dtype=float))
    return seqs, np.concatenate(labs) if labs else np.array([]), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=12_000, help="sequences scanned per group")
    ap.add_argument(
        "--selections",
        default="",
        help="glob of run_al selection files to analyse as well, e.g. '<pools>/sel_*.npz'",
    )
    ap.add_argument("--shard", type=int, default=0, help="this task's index, for array runs")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--cluster-at", type=float, default=0.90)
    ap.add_argument("--min-ic", type=float, default=6.0)
    a = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    from albench.motifs import vocabulary as V

    meme = V._default_meme()
    print(f"PFM source: {meme}", flush=True)
    motifs = V.build(meme=meme, human_only=False, cluster_at=a.cluster_at, min_ic=a.min_ic)
    names = [m.name for m in motifs]
    print(f"vocabulary: {len(motifs)} motifs after clustering at {a.cluster_at}", flush=True)

    rng = np.random.default_rng(0)
    rec: dict = {"source": SOURCE, "sizes": {}, "arms": {}, "selected": {}, "motif_names": names}
    jobs: list[tuple[str, str, Path, list[str]]] = [
        ("sizes", str(sz), POOLS / f"poolA_{sz}_d30000.npz", [f"{SOURCE}={sz}"]) for sz in SIZES
    ]
    # The composition arms: same 300,000 candidates every time, split between one, two,
    # three or four reservoirs. Running the same measurement here is what separates "the
    # rule narrows diversity" from "this one designed reservoir is narrow" -- the pool-size
    # sweep alone cannot tell those apart, because it only ever offers motif_shared_core.
    for p in sorted(POOLS.glob("pool[BRE]_*_d30000_mix.json")):
        arm = p.name.replace("_d30000_mix.json", "")
        srcs = json.loads(p.read_text()).get("sources", [])
        if srcs:
            jobs.append(("arms", arm, POOLS / f"{arm}_d30000.npz", srcs))

    # Every arm run through run_al: the pool-size sweep, the rule replicates, the layer
    # sweep, the 300k start. run_al stamps `acq_from` into the selection, so the candidate
    # pool is read from the file rather than guessed from the tag -- which matters because
    # the tags spell the same pool three different ways (540000, 540k, u6_540k).
    if a.selections:
        for sp in sorted(Path(a.selections).parent.glob(Path(a.selections).name)):
            try:
                zz = np.load(sp, allow_pickle=True)
                src = str(zz["acq_from"]) if "acq_from" in zz.files else ""
            except Exception:
                continue
            cand = sp.parent / src
            if not src or not cand.exists():
                continue
            jobs.append(("selected", sp.name.replace(".npz", ""), sp, ["@" + str(cand)]))

    if a.nshards > 1:
        # Round-robin rather than contiguous blocks: the pool-size arms get steadily more
        # expensive, so a contiguous split would hand one task the 1M pool and leave
        # another idle.
        jobs = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]
        print(f"shard {a.shard}/{a.nshards}: {len(jobs)} arms")

    for bucket, key, sel_p, srcs in jobs:
        if not sel_p.exists():
            print(f"  skip {key}: no {sel_p.name}")
            continue
        pool, pool_y, ok = _rebuild_pool(srcs)
        if not ok:
            print(f"  skip {key}: candidate pool could not be reconstructed from {srcs}")
            continue
        print(f"\n{bucket[:-1]} {key}  ({', '.join(srcs)})")
        zs = np.load(sel_p, allow_pickle=True)
        # A run_al selection is seed prefix + acquired rows; only the acquired part was
        # chosen, so the prefix is dropped or every arm would look like its own seed set.
        # Exactly, from the stamp: run_al records how many rows it acquired, so the prefix
        # is len - acquired. A fixed 30,000 would be right for the 30k-start arms and
        # silently wrong for the 300k ones, where it would treat 270,000 seed rows as
        # selections.
        nb = 0
        if bucket == "selected":
            nb = len(zs["sequences"]) - int(zs["acq_selected"]) if "acq_selected" in zs.files else 0
        sel = [str(s) for s in zs["sequences"][nb:]]
        sel_y = np.asarray(zs["oracle_labels"][nb:], dtype=float)
        pool_y = np.asarray(pool_y, dtype=float)
        chosen = set(sel)
        # The reconstruction is a guess about which rows select_corpus offered, so it is
        # checked rather than trusted: nearly every selected sequence must be found in it.
        # A reconstruction that is subtly wrong would give a "not selected" group that is
        # really a different pool, and every contrast below would be meaningless.
        pool_set = set(pool)
        inpool = sum(1 for s in sel if s in pool_set) if len(pool) < 400_000 else None
        if inpool is not None and inpool < 0.99 * len(sel):
            print(
                f"  SKIP {key}: only {inpool:,}/{len(sel):,} selected rows are in the "
                "reconstructed candidate pool"
            )
            continue
        rest = [i for i, s in enumerate(pool) if s not in chosen]
        print(f"  {len(sel):,} selected | {len(rest):,} not selected | {len(pool):,} in pool")
        if len(pool) == len(sel) and rest:
            raise SystemExit(
                f"{key}: the pool and the selection are the same size but {len(rest):,} pool "
                "rows are not in the selection -- the reconstructed pool is not the one the "
                "arm actually chose from, so every contrast here would be against the wrong "
                "reference"
            )

        def take(xs, ys, n):
            if len(xs) <= n:
                return list(xs), (np.asarray(ys) if ys is not None else None)
            i = rng.choice(len(xs), n, replace=False)
            return [xs[j] for j in i], (np.asarray(ys)[i] if ys is not None else None)

        groups = {"sources": srcs}
        s_x, s_y = take(sel, sel_y, a.sample)
        groups["selected"] = describe(s_x, s_y, motifs, names, device, "selected")
        # How much of the POOL's activity axis the selection actually covers, in the pool's
        # own decile bins. 1.0 means the selection reaches every tenth of the range the
        # pool offered; a rule that creams off the top scores low here while still
        # reporting a high mean.
        try:
            edges = np.quantile(pool_y, np.linspace(0, 1, 11))
            groups["selected"]["pool_decile_coverage"] = round(
                float(len(set(np.clip(np.digitize(s_y, edges[1:-1]), 0, 9))) / 10), 3
            )
        except Exception:
            pass
        if rest:
            r_x, r_y = take([pool[i] for i in rest], pool_y[rest], a.sample)
            groups["not_selected"] = describe(r_x, r_y, motifs, names, device, "not selected")
        p_x, p_y = take(pool, pool_y, a.sample)
        groups["pool"] = describe(p_x, p_y, motifs, names, device, "whole pool")
        rec[bucket][key] = groups

    # The built-in control: at a candidate pool of 30,000 the rule selects all 30,000, so
    # `selected` and `pool` are the same sequences and every metric must agree. They are
    # independently sampled here, so the tolerance is sampling noise, not zero.
    g = rec["sizes"].get("30000")
    if g and "not_selected" in g:
        raise SystemExit(
            "the 30,000-candidate arm has a non-empty 'not selected' group, which means the "
            "selected set is not the whole pool and the no-op control does not hold"
        )
    if g:
        # Compare the SETS, not the metrics. The two groups are sampled independently from
        # the same 30,000 sequences, so their metrics differ by sampling noise -- at n=150
        # the effective motif count moved 28% on identical data, which would fail any
        # metric-based tolerance for reasons that have nothing to do with correctness.
        # Set equality is the actual invariant and does not depend on the sample size.
        d = abs(g["selected"]["effective_motif_count"] - g["pool"]["effective_motif_count"])
        rel = d / max(1e-9, g["pool"]["effective_motif_count"])
        print(
            f"\nno-op control at 30k: effective motif count differs by {rel:.1%} "
            f"(sampling noise on identical sequences; the set check below is the real one)"
        )

    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
