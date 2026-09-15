"""Generate and label one pool, so training sets can be drawn from it by subsetting.

One pool per (strategy, parameters, generation seed). Every training set for that
combination -- every size on the scaling curve, every subset replicate -- is then a
NESTED subset of this pool, so a scaling comparison varies only the amount of data
rather than also varying which sequences were drawn.

Cost, measured 2026-09-15 rather than projected: the AG oracle labels at ~300
sequences/s in steady state, so a 300k pool is ~17 minutes. (An earlier 256-sequence
probe suggested 29.7 seq/s, but that was ~90% one-time JAX compilation; it amortises
away at real batch sizes.)

Usage:
    python scripts/build_pool.py --strategy zoonomia --size 300000 \
        --set rate_mode=per_position --generation-seed 0 --out-dir outputs/pools
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

from albench.pools import PoolSpec, write_pool  # noqa: E402
from albench.registry import Context, get  # noqa: E402

logger = logging.getLogger("build_pool")


def _coerce(v: str):
    low = v.strip().lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--size", type=int, default=300_000)
    ap.add_argument("--generation-seed", type=int, default=0)
    ap.add_argument("--task", default="k562", choices=["k562", "yeast"])
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", default=[])
    ap.add_argument("--out-dir", default="outputs/pools")
    ap.add_argument("--oracle", default="ag_s2")
    ap.add_argument("--oracle-id", default="full856k_clean")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="exit successfully if this pool already exists (makes array resubmission safe)",
    )
    args = ap.parse_args()

    params = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        params[k.strip()] = _coerce(v)

    spec = PoolSpec(
        strategy=args.strategy,
        params=params,
        size=args.size,
        generation_seed=args.generation_seed,
        task=args.task,
    )
    out = spec.path(args.out_dir)
    if args.skip_existing and out.exists() and out.stat().st_size > 0:
        logger.info("SKIP pool already present: %s", out)
        return 0

    logger.info("pool %s", spec.pool_id)
    logger.info(
        "  strategy=%s params=%s size=%d gen_seed=%d",
        spec.strategy,
        params,
        spec.size,
        spec.generation_seed,
    )

    # ---- generate ----------------------------------------------------------
    t0 = time.time()
    sspec = get(args.strategy)
    # Load the genomic pool only for strategies that derive from it, so a purely
    # generative strategy does not pay for reading it.
    pool_seqs = None
    if sspec.needs_pool:
        from albench.paths import resolve

        z = np.load(resolve("bg_cache"), allow_pickle=True)
        pool_seqs = [str(s) for s in z["sequences"]]
    ctx = Context(task=args.task, pool_sequences=pool_seqs)
    seqs, _meta = sspec.generate(spec.size, ctx, seed=spec.generation_seed, **params)
    gen_s = time.time() - t0
    logger.info(
        "  generated %d sequences in %.0fs (%.0f seq/s)",
        len(seqs),
        gen_s,
        len(seqs) / max(gen_s, 1e-6),
    )

    n_unique = len(set(seqs))
    if n_unique < 0.99 * len(seqs):
        # Duplicates waste oracle time and inflate apparent coverage. Warn loudly
        # rather than silently labelling the same sequence many times.
        logger.warning(
            "  WARNING: only %d/%d sequences are unique (%.1f%% duplicates). "
            "Check the strategy's capacity at this size.",
            n_unique,
            len(seqs),
            100 * (1 - n_unique / len(seqs)),
        )

    # ---- label -------------------------------------------------------------
    t0 = time.time()
    from experiments.exp1_1_scaling import _load_oracle
    from scripts.generate_labeled_pools import _label_sequences

    oracle = _load_oracle(args.task, oracle_type=args.oracle)
    logger.info("  oracle loaded in %.0fs", time.time() - t0)

    t0 = time.time()
    labels = _label_sequences(oracle, seqs)
    lab_s = time.time() - t0
    logger.info("  labelled in %.0fs (%.0f seq/s)", lab_s, len(seqs) / max(lab_s, 1e-6))

    write_pool(out, seqs, labels, spec, oracle_id=args.oracle_id)
    logger.info("DONE %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
