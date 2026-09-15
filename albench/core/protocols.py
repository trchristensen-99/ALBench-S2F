"""The five interfaces a new setting has to satisfy.

Nothing here is implemented yet, by design. This file is the contract in
docs/STRUCTURE.md written so it type-checks: it names what a domain must provide, and
each docstring records the mistake the interface exists to prevent. Today's code
already satisfies these shapes informally -- the work of the migration is making that
explicit, not inventing new abstractions.

These are ``typing.Protocol``, so a domain satisfies one by having the right methods.
No base class to inherit, no registration to forget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import pandas as pd


@runtime_checkable
class Oracle(Protocol):
    """Assigns labels to sequences. One study, one oracle."""

    id: str
    """Provenance stamp written into every pool this oracle labels.

    Not decoration. Pools labelled by a non-canonical oracle are indistinguishable
    from good ones once written, and this project has already had to audit its way
    back out of that: `full856k_clean` became the canonical id only after a label
    provenance review found unstamped batteries. Anything that writes labels must
    write this alongside them, and anything that reads labels should refuse a pool
    that lacks it.
    """

    def label(self, seqs: Sequence[str]) -> np.ndarray:
        """Return one label per sequence, in the order given.

        Must return finite values. A silent NaN here becomes a training set that
        looks fine and a student that quietly learns nothing from those rows.
        """
        ...


@runtime_checkable
class Fitted(Protocol):
    """A trained student."""

    def predict(self, seqs: Sequence[str]) -> np.ndarray: ...

    def embed(self, seqs: Sequence[str]) -> np.ndarray | None:
        """Per-sequence representation, or None if this student has none.

        Returning None is a supported answer, not a failure. BADGE and BatchBALD need
        embeddings; a student that cannot supply them should degrade to
        uncertainty-only selection rather than crash -- and the caller must be able to
        tell which of the two happened, because silently switching selection strategy
        would change what an experiment measures.
        """
        ...


@runtime_checkable
class Student(Protocol):
    """Fits a model to (sequences, labels) under a hyperparameter configuration."""

    def fit(self, seqs: Sequence[str], labels: np.ndarray, hp: dict) -> Fitted:
        """Train and return a fitted model.

        `hp` is a plain dict so the HP-search strategies stay orthogonal: searching
        over architectures is a property of the student, not of the reservoir that
        produced the data or the acquisition that selected it. Conflating those is
        why HP search kept leaking into the reservoir abstraction.
        """
        ...


@runtime_checkable
class Reservoir(Protocol):
    """Generates candidate sequences."""

    def generate(self, n: int, ctx: object) -> tuple[list[str], pd.DataFrame]:
        """Return `n` sequences plus a frame of per-sequence provenance.

        The frame is what makes a strategy auditable after the fact -- which peak,
        which motif, which parent sequence, which mutation rate. Several silent bugs
        in this project (duplicate-heavy motif clustering, a dead GC-bin parameter)
        were only visible in that metadata, not in the sequences themselves.
        """
        ...


@runtime_checkable
class EvalSuite(Protocol):
    """Scores a fitted student on whatever test sets the domain cares about."""

    def score(self, fitted: Fitted) -> dict[str, float]:
        """Return {test_set_name: metric}.

        Deliberately a flat dict of floats. The eval sets ARE the scientific content
        and differ completely between domains -- SNV effect sizes here, DREAM eval
        classes there -- so the harness should carry them without claiming to
        understand them. Any richer schema would be invented for a generality nobody
        has asked for.
        """
        ...
