"""Random DNA at a controlled GC content.

Two needs this covers that ``random`` (fixed ~0.5) and ``gc_matched`` (matches a pool's GC
*distribution*) do not: a single reservoir pinned to a chosen GC, and a single reservoir that
spans a *range* of GC evenly. The second is Peter's point that there is no reason to commit a
reservoir to one GC content -- synthesise each GC batch separately (cost is linear in the number
of levels) and pool them. Composition is controlled per base only; this introduces no
dinucleotide or motif structure, which is the point of a composition baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from albench.reservoir.base import ReservoirSampler

# integer 0-3 -> nucleotide byte, in the order the per-base GC probabilities below assume.
_NUC_BYTES = np.frombuffer(b"ACGT", dtype=np.uint8)


class GCRandomSampler(ReservoirSampler):
    """Generate i.i.d. random DNA with a target GC fraction.

    ``target_gc`` fixes GC for every sequence. ``gc_levels`` (when set) overrides it: the pool
    is split into equal batches, one per level, so the reservoir spans a controlled GC range.
    Each base is drawn independently with P(G)=P(C)=gc/2 and P(A)=P(T)=(1-gc)/2.
    """

    def __init__(
        self,
        seed: int | None = None,
        target_gc: float = 0.65,
        gc_levels: str | list[float] | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self.target_gc = float(target_gc)
        if gc_levels is None or gc_levels == "":
            self.gc_levels: list[float] | None = None
        elif isinstance(gc_levels, str):
            self.gc_levels = [float(x) for x in gc_levels.split(",") if x.strip()]
        else:
            self.gc_levels = [float(x) for x in gc_levels]

    def sample(
        self,
        candidates: list[str],
        n_samples: int,
        metadata: list[dict[str, object]] | None = None,
    ) -> list[int]:
        """Sample indices uniformly without replacement."""
        if n_samples > len(candidates):
            raise ValueError("n_samples cannot exceed number of candidates")
        return self._rng.choice(len(candidates), size=n_samples, replace=False).tolist()

    def _gen_at_gc(self, n: int, seq_len: int, gc: float, batch: int = 200_000) -> list[str]:
        """Vectorised generation of ``n`` sequences at a single GC, batched to bound memory."""
        gc = min(max(gc, 0.0), 1.0)
        p = np.array([(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2])  # A C G T
        out: list[str] = []
        for start in range(0, n, batch):
            nb = min(batch, n - start)
            idx = self._rng.choice(4, size=(nb, seq_len), p=p).astype(np.uint8)
            flat = _NUC_BYTES[idx.ravel()].reshape(nb, seq_len)
            out.extend(row.tobytes().decode("ascii") for row in flat)
        return out

    def generate(
        self,
        n_sequences: int,
        task: str = "k562",
        batch_size: int = 200_000,
    ) -> tuple[list[str], pd.DataFrame]:
        """Generate ``n_sequences`` random sequences at the configured GC (or GC sweep)."""
        seq_len = 200 if task == "k562" else 80
        if self.gc_levels:
            levels = self.gc_levels
            counts = [n_sequences // len(levels)] * len(levels)
            for i in range(n_sequences - sum(counts)):  # spread the remainder
                counts[i] += 1
            seqs: list[str] = []
            gc_tags: list[float] = []
            for gc, c in zip(levels, counts):
                if c <= 0:
                    continue
                seqs.extend(self._gen_at_gc(c, seq_len, gc, batch_size))
                gc_tags.extend([gc] * c)
            order = self._rng.permutation(len(seqs))  # interleave the batches in the pool
            seqs = [seqs[i] for i in order]
            gc_tags = [gc_tags[i] for i in order]
            method = "gc_even_sweep"
        else:
            seqs = self._gen_at_gc(n_sequences, seq_len, self.target_gc, batch_size)
            gc_tags = [self.target_gc] * len(seqs)
            method = f"gc_fixed_{self.target_gc:.2f}"

        if task == "yeast":
            from albench.reservoir.random_sampler import _YEAST_FLANK_3, _YEAST_FLANK_5

            seqs = [_YEAST_FLANK_5 + s + _YEAST_FLANK_3 for s in seqs]

        meta = pd.DataFrame(
            {
                "seq_idx": np.arange(len(seqs), dtype=np.int64),
                "target_gc": np.asarray(gc_tags, dtype=np.float32),
                "method": method,
                "source": "generated",
            }
        )
        return seqs, meta
