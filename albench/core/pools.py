"""Label once into a large pool, then draw nested training sets from it.

WHY. Oracle labelling dominates the cost of everything: the AG oracle runs 20 forward
passes per sequence (10 folds x reverse-complement averaging) through a 551M-parameter
backbone, giving ~30 sequences/second. At that rate a 300k cell is ~2.8 hours.

The screen as originally expanded labelled every CELL independently -- 35 parameter
combinations x 2 sizes x 3 seeds = 210 separate labelling jobs, ~323 GPU-h, even
though cells of the same strategy and parameters differ only in how many sequences
they use and which. Labelling one large pool per (strategy, parameters) and drawing
training sets from it costs ~98 GPU-h instead, for exactly the same experiments.

IT IS ALSO BETTER SCIENCE, not merely cheaper. Subsets drawn here are NESTED: the
D=30k set is a strict subset of the D=100k set, which is a strict subset of the pool.
A scaling curve built from nested subsets varies only the amount of data, whereas
independently generated sets at each size also vary WHICH sequences were drawn, and
that confound is impossible to separate from the scaling effect afterwards.

WHAT NESTING DOES NOT GIVE YOU. Two subsets of one pool differ by sampling only. They
do not capture variation from re-running the generator, which for these strategies is
real -- the multi-seed work measured a reservoir-seed effect of about +0.006 in r,
and larger on some reservoirs. So:

  screen / parameter comparison   subset seeds are fine, and 3x cheaper
  headline scaling numbers        use several independently GENERATED pools

``n_generation_seeds`` in the pool spec is that dial: it is the number of times the
generator is re-run, as distinct from the number of subsets drawn from each pool.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

POOL_SCHEMA = 1


@dataclass
class PoolSpec:
    """One labelled pool: a strategy at fixed parameters, generated once."""

    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    size: int = 300_000
    generation_seed: int = 0
    task: str = "k562"

    @property
    def pool_id(self) -> str:
        """Stable identifier derived from everything that affects the content."""
        payload = json.dumps(
            {
                "strategy": self.strategy,
                "params": {k: str(v) for k, v in sorted(self.params.items())},
                "size": self.size,
                "seed": self.generation_seed,
                "task": self.task,
                "schema": POOL_SCHEMA,
            },
            sort_keys=True,
        )
        digest = hashlib.sha1(payload.encode()).hexdigest()[:10]
        return f"{self.strategy}__g{self.generation_seed}__{digest}"

    def path(self, root: str | Path) -> Path:
        return Path(root) / f"{self.pool_id}.npz"


def write_pool(
    path: str | Path,
    sequences: list[str],
    labels: np.ndarray,
    spec: PoolSpec,
    oracle_id: str,
) -> Path:
    """Write a labelled pool atomically, with enough provenance to trust it later."""
    labels = np.asarray(labels, dtype=np.float32).ravel()
    if labels.shape[0] != len(sequences):
        raise ValueError(f"{len(sequences)} sequences but {labels.shape[0]} labels")
    if not np.isfinite(labels).all():
        raise ValueError("pool contains non-finite labels; refusing to write")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        sequences=np.array(sequences, dtype=object),
        oracle_labels=labels,
        pool_id=spec.pool_id,
        strategy=spec.strategy,
        params=json.dumps(spec.params, default=str),
        generation_seed=spec.generation_seed,
        task=spec.task,
        oracle_id=oracle_id,
        schema=POOL_SCHEMA,
    )
    tmp.replace(path)  # atomic: a killed job cannot leave a half-written pool
    logger.info(
        "wrote pool %s: %d sequences, label mean %.3f sd %.3f",
        spec.pool_id,
        len(sequences),
        float(labels.mean()),
        float(labels.std()),
    )
    return path


def load_pool(path: str | Path) -> dict[str, Any]:
    """Load a pool and check it carries the provenance we rely on."""
    z = np.load(path, allow_pickle=True)
    missing = {"sequences", "oracle_labels", "pool_id", "oracle_id"} - set(z.files)
    if missing:
        raise ValueError(
            f"{path} is missing {sorted(missing)}. It was probably written before the "
            f"pool format existed; regenerate it rather than guessing its provenance."
        )
    out = {
        "sequences": [str(s) for s in z["sequences"]],
        "labels": np.asarray(z["oracle_labels"], dtype=np.float32).ravel(),
        "pool_id": str(z["pool_id"]),
        "oracle_id": str(z["oracle_id"]),
        "strategy": str(z["strategy"]) if "strategy" in z.files else "",
        "params": json.loads(str(z["params"])) if "params" in z.files else {},
    }
    if len(out["sequences"]) != out["labels"].shape[0]:
        raise ValueError(f"{path}: sequence/label length mismatch")
    return out


def nested_subsets(
    n_pool: int,
    sizes: list[int] | tuple[int, ...],
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Index sets for each size, each a strict subset of every larger one.

    Built by permuting the pool once and taking prefixes, so nesting holds by
    construction rather than by luck. Varying ``seed`` gives a different draw; those
    are SAMPLING replicates, not generation replicates (see the module docstring).
    """
    sizes = sorted(set(int(s) for s in sizes))
    if not sizes:
        raise ValueError("no sizes requested")
    if sizes[-1] > n_pool:
        raise ValueError(
            f"largest requested subset ({sizes[-1]:,}) exceeds the pool ({n_pool:,}). "
            f"Generate a larger pool, or request smaller training sets."
        )
    order = np.random.default_rng(seed).permutation(n_pool)
    return {s: np.sort(order[:s]) for s in sizes}


def verify_nesting(subsets: dict[int, np.ndarray]) -> None:
    """Raise unless every smaller subset is contained in every larger one."""
    sizes = sorted(subsets)
    for small, large in zip(sizes, sizes[1:]):
        a, b = set(subsets[small].tolist()), set(subsets[large].tolist())
        if not a <= b:
            raise AssertionError(
                f"subset {small} is not contained in {large}: "
                f"{len(a - b)} indices differ. Scaling comparisons built on these "
                f"would confound amount-of-data with which-data."
            )


def draw_training_set(
    pool: dict[str, Any],
    size: int,
    seed: int = 0,
    sizes_for_nesting: list[int] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Draw one training set from a pool.

    ``sizes_for_nesting`` should list every size in the scaling curve, so the returned
    set is the correct prefix of the shared permutation and therefore nests with the
    others. Passing only the size you want would give a valid sample but would break
    nesting across the curve.
    """
    n = len(pool["sequences"])
    sizes = sizes_for_nesting or [size]
    if size not in sizes:
        sizes = sorted(set(sizes) | {size})
    idx = nested_subsets(n, sizes, seed=seed)[size]
    seqs = [pool["sequences"][i] for i in idx]
    return seqs, pool["labels"][idx], idx


def pool_plan(
    cells: list[Any],
    pool_size: int = 300_000,
    n_generation_seeds: int = 1,
) -> list[PoolSpec]:
    """Collapse screen cells into the pools that would serve them.

    Cells differing only in D or in subset seed share a pool, which is where the
    saving comes from.
    """
    seen: dict[tuple, PoolSpec] = {}
    for c in cells:
        key = (c.reservoir, tuple(sorted((k, str(v)) for k, v in c.params.items())))
        for g in range(n_generation_seeds):
            k2 = (*key, g)
            if k2 not in seen:
                seen[k2] = PoolSpec(
                    strategy=c.reservoir,
                    params=dict(c.params),
                    size=pool_size,
                    generation_seed=g,
                )
    return list(seen.values())


def describe_saving(
    cells: list[Any],
    pools: list[PoolSpec],
    seq_per_s: float = 29.7,
) -> str:
    """Compare per-cell labelling against pooled labelling, in GPU-hours."""
    per_cell = sum(c.d for c in cells) / seq_per_s / 3600
    pooled = sum(p.size for p in pools) / seq_per_s / 3600
    lines = [
        f"labelling {len(cells)} cells independently : {per_cell:8.1f} GPU-h",
        f"labelling {len(pools)} pools and subsetting : {pooled:8.1f} GPU-h",
        f"saving                                     : {per_cell - pooled:8.1f} GPU-h "
        f"({100 * (1 - pooled / per_cell):.0f}%)",
        f"  at {seq_per_s:.1f} seq/s, which is 20 model passes per sequence "
        f"(10 folds x RC averaging)",
    ]
    return "\n".join(lines)
