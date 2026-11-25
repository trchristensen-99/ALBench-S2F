"""Which SUBSET of each strategy does an acquisition rule actually take?

fig18 answers this one level up -- "uncertainty took 20.7% motif_shared_core" -- and the
collaborators asked the finer question: within EvoAug, is it the translocations or the
point mutations; within the motif reservoirs, how many sites and in what arrangement;
among the model-guided generators, Evo2 or D3 and at which step count.

THE PROVENANCE PROBLEM, and why this script is written the way it is. Every sampler's
``generate()`` already returns a per-sequence metadata table -- parent index, event types,
point-mutation count, planted motifs. The pool writer kept only a scalar ``strategy``
stamp, so the stored pools cannot answer the question at all:

    >>> list(np.load("evoaug_paper2023_d30000.npz").keys())
    ['sequences', 'oracle_labels', 'oracle_id', 'strategy', 'seed']

The fix is not to guess from the sequence. Reservoir generation is seeded and the pools
record their seed, so the table can be REGENERATED -- and then the regenerated sequences
must come back byte-identical to the stored ones. That check is the whole point: if it
passes, every attribute is the one that actually produced the stored row; if it fails,
this script reports the reservoir as unrecoverable rather than attributing selections
from a table that describes different sequences. A near-miss is not partial credit here.

    python scratchpad/nov25/selection_subsets.py \
        --arm outputs/acq52/pools/poolB_00_d60000.npz --out outputs/analysis/subsets.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("ALBENCH_REPO", Path(__file__).resolve().parents[2]))
NEST = REPO / "outputs/grid_pools/k562_nested"

# Metadata columns worth breaking a selection down by, and the label to print. Anything
# else the sampler returns (seq_idx, method, source) identifies the row rather than
# describing it, so it is not a factor.
# Columns that identify a row rather than describe it. Everything else the sampler
# returns is a factor -- an allowlist was tried first and silently dropped real columns
# (motif_planted_v2 calls its site count `n_motifs`, evoaug_structural calls its op list
# `event_types`), which read as "this sampler records nothing".
NOT_FACTORS = {"seq_idx", "source", "method", "parent_idx", "region_idx", "positions"}
MAX_LEVELS = 24  # beyond this a factor is a near-identifier; numerics are binned instead


def regenerate(res: str, n: int, seed: int, task: str = "k562"):
    """(sequences, per-sequence attribute dict) for a reservoir, or None if it cannot run."""
    from albench.cli import _load_pool
    from albench.registry import Context, get

    try:
        spec = get(res)
    except Exception as e:
        print(f"  {res}: not in the registry ({type(e).__name__}: {e})")
        return None
    pool, labels = _load_pool(task, None) if spec.needs_pool else (None, None)
    ctx = Context(task=task, pool_sequences=pool, pool_labels=labels)
    try:
        # spec.generate, NOT the CLI path. `albench generate` runs _dedupe_to_target after
        # this call, which can top up with further batches and returns sequences only --
        # the metadata table would no longer line up with the rows it describes. The
        # position check in main() is what catches a pool that was built with a top-up.
        seqs, meta = spec.generate(n, ctx, seed=seed)
    except Exception as e:
        print(f"  {res}: generate failed ({type(e).__name__}: {e})")
        return None
    return list(map(str, seqs)), _attrs(meta, len(seqs))


def _attrs(meta, n: int) -> dict[str, np.ndarray]:
    """Normalise whatever the sampler returned into {factor: (n,) array}."""
    out: dict[str, np.ndarray] = {}
    if meta is None:
        return out
    cols = {}
    if hasattr(meta, "columns"):  # pandas DataFrame
        cols = {c: meta[c].to_numpy() for c in meta.columns}
    elif isinstance(meta, dict):
        cols = {k: np.asarray(v) for k, v in meta.items() if np.ndim(v) == 1}
    elif isinstance(meta, (list, tuple)) and meta and isinstance(meta[0], dict):
        keys = set().union(*(d.keys() for d in meta))
        cols = {k: np.array([d.get(k) for d in meta], dtype=object) for k in keys}
    for k, v in cols.items():
        if k in NOT_FACTORS or len(v) != n:
            continue
        uniq = len(set(map(str, v[: min(n, 20_000)])))
        if uniq < 2:
            continue  # a constant: it describes the sampler, not this sequence
        if uniq > MAX_LEVELS:
            if np.issubdtype(np.asarray(v).dtype, np.number):
                # Quartile bins, so a continuous knob (mutation rate, phyloP) still has
                # levels a selection can be enriched in.
                q = np.nanquantile(np.asarray(v, dtype=float), [0.25, 0.5, 0.75])
                v = np.array(
                    [f"q{int(np.searchsorted(q, x)) + 1}" for x in np.asarray(v, dtype=float)],
                    dtype=object,
                )
            else:
                continue  # free-form text (motif name lists): not a factor with levels
        out[k] = np.asarray(v)
    return out


def enrichment(sel: np.ndarray, pool: np.ndarray) -> dict:
    """Selected share vs pool share for every level of one factor.

    Reported as a RATIO, because the shares themselves are not comparable between factors
    with different numbers of levels: a factor with 5 levels has a 0.20 baseline and one
    with 2 has 0.50, and a bare percentage would make the first look neglected.
    """
    cs, cp = Counter(map(str, sel)), Counter(map(str, pool))
    ns, np_ = max(1, len(sel)), max(1, len(pool))
    rows = {}
    for lvl in sorted(cp):
        ps, pp = cs.get(lvl, 0) / ns, cp[lvl] / np_
        rows[lvl] = {
            "pool_share": round(pp, 4),
            "selected_share": round(ps, 4),
            "ratio": round(ps / pp, 3) if pp else None,
            "n_selected": cs.get(lvl, 0),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="a pool written by run_al (the SELECTED rows)")
    ap.add_argument("--seed-size", type=int, default=30_000)
    ap.add_argument("--candidates", default="", help="the pool it chose FROM; default from meta")
    ap.add_argument("--gen-seed", type=int, default=42)
    ap.add_argument("--max-regen", type=int, default=1_000_000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    za = np.load(a.arm, allow_pickle=True)
    sel_seq = np.asarray(za["sequences"])[a.seed_size :]
    print(
        f"{Path(a.arm).name}: {len(sel_seq):,} acquired rows (seed prefix of {a.seed_size:,} dropped)"
    )

    # Reservoir level first: it needs nothing but the pool's own source array, and it is
    # the number fig18 reports, recomputed here so the two can be checked against
    # each other rather than quoted from different scripts.
    report: dict = {"arm": str(a.arm), "n_selected": int(len(sel_seq))}
    # Two shapes of arm exist. run_al writes the seed prefix plus the acquired rows and
    # leaves the candidate pool on disk; select_corpus builds its candidate pool in memory
    # and writes only the selection, with a `source` column and a sidecar naming how many
    # candidates each reservoir offered. Read whichever is present -- the baseline a
    # selection is compared against must be the pool that was ACTUALLY offered, and
    # assuming an equal split when it was not is how a rule gets credited for a quota.
    side = Path(str(a.arm).replace(".npz", "_mix.json"))
    if side.exists():
        sc = json.loads(side.read_text())
        offered = {}
        for s in sc.get("sources", []):
            r, _, n = str(s).partition("=")
            offered[r] = int(n)
        tot = sum(offered.values()) or 1
        # select_corpus does not write a per-row source column, only these totals. They
        # are counts of the same selection, so they serve for the reservoir level; the
        # per-row attribution below is rebuilt by matching sequences, not assumed.
        if "source" in za:
            cs = Counter(map(str, np.asarray(za["source"])[a.seed_size :]))
        else:
            cs = Counter({k: int(v) for k, v in sc.get("realised_mixture", {}).items()})
        report["reservoir"] = {
            r: {
                "pool_share": round(n / tot, 4),
                "selected_share": round(cs.get(r, 0) / max(1, sum(cs.values())), 4),
                "ratio": round((cs.get(r, 0) / max(1, sum(cs.values()))) / (n / tot), 3)
                if n
                else None,
                "n_selected": cs.get(r, 0),
            }
            for r, n in offered.items()
        }
        for r, d in sorted(report["reservoir"].items(), key=lambda kv: -kv[1]["selected_share"]):
            print(
                f"    {r:24s} pool {d['pool_share']:.3f} -> selected {d['selected_share']:.3f}"
                f"  ({d['ratio']}x)"
            )

    cand = a.candidates
    if not cand:
        meta = za.get("acquired_from")
        cand = str(meta) if meta is not None else ""
    if "reservoir" not in report and cand and os.path.exists(cand):
        zc = np.load(cand, allow_pickle=True)
        if "source" in zc:
            csrc = np.asarray(zc["source"])[a.seed_size :]
            cseq = np.asarray(zc["sequences"])[a.seed_size :]
            where = {s: i for i, s in enumerate(map(str, cseq))}
            idx = np.array([where[s] for s in map(str, sel_seq) if s in where])
            print(
                f"  matched {len(idx):,}/{len(sel_seq):,} acquired rows back to the candidate pool"
            )
            report["reservoir"] = enrichment(csrc[idx], csrc)
            for r, d in sorted(
                report["reservoir"].items(), key=lambda kv: -kv[1]["selected_share"]
            ):
                print(
                    f"    {r:24s} pool {d['pool_share']:.3f} -> selected {d['selected_share']:.3f}"
                    f"  ({d['ratio']}x)"
                )

    # Within-reservoir level: regenerate, VERIFY, then attribute.
    report["within"] = {}
    for res in sorted(report.get("reservoir", {})):
        store = NEST / f"{res}_d1000000.npz"
        if not store.exists():
            cands = sorted(NEST.glob(f"{res}_d*.npz"), key=lambda p: p.stat().st_size)
            if not cands:
                continue
            store = cands[-1]
        zp = np.load(store, allow_pickle=True)
        pool_seq = np.asarray(zp["sequences"])
        n = min(len(pool_seq), a.max_regen)
        seed = int(zp["seed"].item()) if "seed" in zp else a.gen_seed
        print(f"  regenerating {res} (n={n:,}, seed={seed}) to recover its attribute table")
        got = regenerate(res, n, seed)
        if got is None:
            report["within"][res] = {"status": "unrecoverable: generate failed"}
            continue
        gen_seq, attrs = got
        # Positions spread over the whole range, not the first k. `albench generate` tops
        # up duplicates by appending further batches, which shifts every row after the
        # first removal; a head-only check would pass on a pool whose tail is misaligned
        # and then attribute thousands of selections to the wrong generation parameters.
        probe = np.unique(np.linspace(0, min(len(gen_seq), len(pool_seq)) - 1, 2000).astype(int))
        k = len(probe)
        same = sum(1 for i in probe if gen_seq[i] == str(pool_seq[i]))
        if same < k:
            # Not "close enough". A table that describes different sequences would attach
            # every attribute to the wrong row, and the output would look perfectly normal.
            report["within"][res] = {
                "status": f"unrecoverable: regenerated sequences differ ({same}/{k} match)"
            }
            print(f"    REGENERATION DOES NOT MATCH the stored pool ({same}/{k}) -- not attributed")
            continue
        if not attrs:
            report["within"][res] = {"status": "verified, but the sampler records no factors"}
            print("    verified, but this sampler returns no per-sequence factors")
            continue
        print(f"    verified on {k} rows; factors: {', '.join(sorted(attrs))}")
        where = {s: i for i, s in enumerate(gen_seq)}
        hit = np.array([where[s] for s in map(str, sel_seq) if s in where])
        if len(hit) < 50:
            report["within"][res] = {"status": f"only {len(hit)} selected rows traced"}
            continue
        report["within"][res] = {"status": "ok", "n_traced": int(len(hit)), "factors": {}}
        for f, v in attrs.items():
            report["within"][res]["factors"][f] = enrichment(v[hit], v)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
