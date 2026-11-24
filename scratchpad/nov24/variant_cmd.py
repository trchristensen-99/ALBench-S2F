"""Emit one variant's build arguments, one per line, for the sbatch array to read.

The array task needs four things: the readable alias, the strategy, the --set arguments,
and the exact path build_pool.py will write. The last one is why this exists rather than
a shell one-liner -- the canonical pool name is a sha1 of the parameters, so the only
safe way to find the file afterwards is to ask PoolSpec for it. The obvious alternative,
``ls -t | head -1``, races with the other 24 tasks writing into the same directory and
would hard-link the wrong pool under a readable name, which is the worst possible
failure here: a plausible, provenance-stamped file under a name that lies.

    python scratchpad/nov24/variant_cmd.py 7 --field path
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from variants import GUIDES, VARIANTS  # noqa: E402

RAW_DIR = "outputs/grid_pools/k562_variants_raw"
SIZE = 30_000
GEN_SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int)
    ap.add_argument(
        "--field",
        default="all",
        choices=("all", "name", "strategy", "setargs", "path", "why", "guide"),
    )
    ap.add_argument("--count", action="store_true", help="print how many variants exist")
    a = ap.parse_args()
    if a.count:
        print(len(VARIANTS))
        return
    if not 0 <= a.index < len(VARIANTS):
        raise SystemExit(f"index {a.index} out of range: {len(VARIANTS)} variants")
    name, strategy, params, why = VARIANTS[a.index]

    from albench.core.pools import PoolSpec
    from albench.core.registry import get

    # The guide is part of the pool's identity, exactly as build_pool.py records it:
    # two pools that differ only in which model selected their sequences are different
    # pools, and must not share a content hash.
    guide = GUIDES.get(name)
    ident = dict(params)
    if "oracle" in (get(strategy).assets or ()):
        ident["guide_model"] = f"bank:{os.path.basename(guide)}" if guide else "oracle:ag_s2"

    spec = PoolSpec(
        strategy=strategy, params=ident, size=SIZE, generation_seed=GEN_SEED, task="k562"
    )
    fields = {
        "name": name,
        "strategy": strategy,
        "guide": guide or "",
        # One flag per line so an empty parameter set emits nothing at all, rather than
        # an empty string that the shell would pass as a positional argument.
        "setargs": "\n".join(f"--set\n{k}={v}" for k, v in sorted(params.items())),
        "path": str(spec.path(RAW_DIR)),
        "why": why,
    }
    if a.field == "all":
        for k in ("name", "strategy", "path", "why"):
            print(f"{k}\t{fields[k]}")
        return
    out = fields[a.field]
    if out:
        print(out)


if __name__ == "__main__":
    main()
