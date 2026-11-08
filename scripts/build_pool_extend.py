"""Grow an existing canonical pool to a larger size without breaking the prefix invariant.

`build_pool.py --size 1000000` would draw a fresh 1M sequences, and a fresh draw is NOT a prefix-superset of
the 300k pool already on disk -- measured Nov 7 2026, only `motif_shared_core` happened to be prefix-stable and
every other reservoir was not. Rewriting the smaller sizes to match would invalidate every number already
produced on them, so instead this keeps the existing pool verbatim and appends sequences the draw has that the
pool does not. `seqs[:300000]` therefore stays byte-identical, and the size axis stays a size axis.

Only the appended sequences are sent to the oracle, which is the whole saving: extending 300k -> 1M costs 700k
labels rather than 1M.

    python scripts/build_pool_extend.py --strategy motif_shared_core --target 1000000 \
        --base outputs/grid_pools/k562_nested/motif_shared_core_d300000.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

logger = logging.getLogger("build_pool_extend")

# The draw has to cover the whole TARGET, not just the shortfall: a sampler restarts its stream from the
# beginning for a given generation seed, so asking it for `need` sequences hands back the first `need` of
# the same stream the base pool already contains -- measured on motif_shared_core, where a 1,200-sequence
# draw was 100% already present. Headroom on top absorbs collisions. Generation is cheap next to labelling
# and the surplus is discarded BEFORE the oracle sees it, so this costs minutes, not GPU-hours.
HEADROOM = 1.2


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--base", required=True, help="existing pool, kept verbatim as the prefix")
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--out", help="defaults to the base path with the new size in its name")
    ap.add_argument("--generation-seed", type=int, default=42)
    ap.add_argument("--task", default="k562")
    ap.add_argument("--oracle", default="ag_s2")
    ap.add_argument("--set", action="append", default=[], help="strategy parameter, key=value")
    a = ap.parse_args()

    base_path = REPO / a.base if not Path(a.base).is_absolute() else Path(a.base)
    z = np.load(base_path, allow_pickle=True)
    base_seqs = [str(s) for s in z["sequences"]]
    base_y = z["oracle_labels"].astype(np.float32)
    oracle_id = str(z["oracle_id"])
    need = a.target - len(base_seqs)
    out = Path(a.out) if a.out else base_path.with_name(f"{a.strategy}_d{a.target}.npz")
    if need <= 0:
        raise SystemExit(f"base already holds {len(base_seqs):,}, which is >= target {a.target:,}")
    logger.info("base %s: %d sequences; need %d more", base_path.name, len(base_seqs), need)
    if out.exists() and out.stat().st_size > 0:
        logger.info("SKIP already present: %s", out)
        return 0

    params = {}
    for item in a.set:
        k, v = item.split("=", 1)
        params[k.strip()] = int(v) if v.isdigit() else v

    from albench.registry import Context, get

    sspec = get(a.strategy)
    pool_seqs = None
    if sspec.needs_pool:
        from albench.paths import resolve

        pz = np.load(resolve("bg_cache"), allow_pickle=True)
        pool_seqs = [str(s) for s in pz["sequences"]]
    ctx = Context(task=a.task, pool_sequences=pool_seqs)

    t0 = time.time()
    draw = int(a.target * HEADROOM)
    seqs, _ = sspec.generate(draw, ctx, seed=a.generation_seed, **params)
    logger.info("generated %d in %.0fs", len(seqs), time.time() - t0)
    # A sampler whose stream is prefix-stable reproduces the base pool exactly, and the extension is then
    # just the tail of one long draw; one that is not gives a scrambled draw we have to filter. Which one
    # it is changes nothing downstream, but it is worth knowing which reservoir behaves how.
    stable = len(seqs) >= len(base_seqs) and all(
        str(seqs[i]) == base_seqs[i] for i in (0, 1, len(base_seqs) // 2, len(base_seqs) - 1)
    )
    logger.info("draw is %s of the base pool", "a prefix-superset" if stable else "NOT a superset")

    have = set(base_seqs)
    novel: list[str] = []
    for s in seqs:
        s = str(s)
        if s in have:
            continue
        have.add(s)  # also drops repeats WITHIN the draw, so the extension is itself deduplicated
        novel.append(s)
        if len(novel) == need:
            break
    logger.info(
        "kept %d novel of %d drawn (%.1f%% were already in the base or repeats of each other)",
        len(novel),
        len(seqs),
        100 * (1 - len(novel) / max(1, len(seqs))),
    )
    if len(novel) < need:
        raise SystemExit(
            f"only {len(novel):,} novel sequences from a {draw:,} draw; the reservoir cannot reach "
            f"{a.target:,} distinct at these parameters. Raise HEADROOM or lower the target."
        )

    from experiments.exp1_1_scaling import _load_oracle
    from scripts.generate_labeled_pools import _label_sequences

    t0 = time.time()
    oracle = _load_oracle(a.task, oracle_type=a.oracle)
    logger.info("oracle loaded in %.0fs", time.time() - t0)
    t0 = time.time()
    y_new = np.asarray(_label_sequences(oracle, novel), dtype=np.float32)
    logger.info(
        "labelled %d in %.0fs (%.0f seq/s)",
        len(novel),
        time.time() - t0,
        len(novel) / max(time.time() - t0, 1e-6),
    )

    all_seqs = base_seqs + novel
    all_y = np.concatenate([base_y, y_new])
    assert len(all_seqs) == a.target and len(set(all_seqs)) == a.target, "extension is not distinct"
    tmp = out.with_suffix(".tmpnew.npz")
    np.savez(
        tmp,
        sequences=np.asarray(all_seqs, dtype=object),
        oracle_labels=all_y,
        oracle_id=oracle_id,
        strategy=a.strategy,
        seed=np.int64(a.generation_seed),
        extended_from=str(base_path.name),
        n_base=np.int64(len(base_seqs)),
    )
    tmp.replace(out)
    logger.info(
        "DONE %s | %d = %d base + %d new | label %+.3f+-%.3f (base %+.3f, new %+.3f)",
        out,
        a.target,
        len(base_seqs),
        len(novel),
        all_y.mean(),
        all_y.std(),
        base_y.mean(),
        y_new.mean(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
