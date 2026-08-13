"""FiNeMo-based motif reservoir (the meeting's REVISE of motif_planted, to avoid pushback).

Pipeline: FiNeMo identifies motif instances in real sequences -> EXCISE the motif spans ->
dinuc-shuffle the remaining background (destroys latent grammar, keeps composition) -> RE-INSERT
the motifs (in their genomic context position, OR at random positions). Builds directly on the
existing albench/reservoir/tf_motif_shuffle.py (TFMotifShuffleSampler already does excise+shuffle).

STATUS: scaffold. Needs: (1) FiNeMo motif-instance calls (hits: seq_id, start, end, motif_id,
strand) — run FiNeMo on the CWM/attribution set, or load precomputed hits; (2) place-back mode.
"""

import numpy as np

_BASES = np.array(list("ACGT"))


def _dinuc_shuffle(seq: str, rng) -> str:
    """Altschul-Erickson-style dinucleotide-preserving shuffle (reuse the repo's implementation)."""
    from albench.reservoir.tf_motif_shuffle import dinuc_shuffle  # existing helper

    return dinuc_shuffle(seq, rng)


def build_finemo_motif_sequence(seq: str, motif_hits, rng, place: str = "genomic_context") -> str:
    """seq: real sequence. motif_hits: list of (start, end) motif spans in seq (from FiNeMo).
    place: 'genomic_context' (re-insert at original positions) or 'random' (random positions)."""
    spans = sorted(motif_hits)
    motifs = [seq[s:e] for s, e in spans]
    # 1. excise motifs -> background
    keep, prev = [], 0
    for s, e in spans:
        keep.append(seq[prev:s])
        prev = e
    keep.append(seq[prev:])
    background = "".join(keep)
    # 2. dinuc-shuffle the background
    shuffled_bg = _dinuc_shuffle(background, rng)
    # 3. re-insert motifs
    if place == "genomic_context":
        # rebuild at the original coordinates (map old positions into the shuffled bg of equal length
        # minus motif spans) — simplest: re-insert at the same offsets
        out = list(shuffled_bg)
        for (s, e), m in zip(spans, motifs):
            out[s:s] = list(m)
        return "".join(out)[: len(seq)]
    elif place == "random":
        out = list(shuffled_bg)
        for m in motifs:
            pos = int(rng.integers(0, len(out) + 1))
            out[pos:pos] = list(m)
        return "".join(out)[: len(seq)]
    raise ValueError(place)


# TODO(sampler): wrap as a ReservoirSampler subclass registered as 'motif_finemo', taking a FiNeMo
# hits table (precomputed) + place mode; oracle-label downstream like the other reservoir caches.
# Then add to the exp1_1_scaling.py reservoir dispatch + generate_reservoir_cache prefixes.
