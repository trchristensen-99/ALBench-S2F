"""Cell-type conditioning: one trunk, many cell types, and a way to reach a cell never trained on.

The study's students and oracles have so far been per-cell: a K562 model and a HepG2 model,
each blind to the other. That cannot answer the question the scaling work is for -- which
training data to prioritise *for a given cell type* -- and it cannot reach a cell type with
no labels at all.

This package holds the conditioning schemes and the controls that tell them apart:

``conditioning``  how a cell identity enters an architecture (FiLM, at the readout or
                  throughout the trunk), for every scheme from a free embedding to a frozen
                  text description.
``descriptions``  what we say about each cell line, and the frozen encoder that turns it
                  into a vector.
``data``          the long-format (sequence, cell, label) view the trainer consumes.

The reason the package exists rather than a flag on the menu trainer: a conditioning scheme
is only interesting next to its controls, and the controls (a cell-agnostic model, a
description-shuffled model) have to be constructed the same way or the comparison is not a
comparison. Keeping them in one place is what makes it hard to report the method without
them -- the same reason BADGE ships with ``badge_epistemic_only``.
"""

from __future__ import annotations

from albench.celltype.conditioning import CONDITIONING_MODES, ConditionedModel, wrap
from albench.celltype.descriptions import CELL_LINES, describe, text_embeddings

__all__ = [
    "CELL_LINES",
    "CONDITIONING_MODES",
    "ConditionedModel",
    "describe",
    "text_embeddings",
    "wrap",
]
