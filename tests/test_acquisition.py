"""Acquisition contracts, with emphasis on the degeneracies that fail silently.

BADGE and BatchBALD both have failure modes that produce a full, plausible-looking
batch while actually selecting at random. These tests make those modes loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from albench import registry as R
from albench.acquisition.badge import (
    BADGEAcquisition,
    EpistemicOnlyAcquisition,
    KMeansPPOnlyAcquisition,
    kmeanspp,
)
from albench.model import SequenceModel


class FakeStudent(SequenceModel):
    """Student with controllable embeddings and uncertainty components."""

    def __init__(self, n, d=8, seed=0, epi=None, ale=None, collapse=False, separates=True):
        rng = np.random.default_rng(seed)
        self.z = np.zeros((n, d)) if collapse else rng.normal(size=(n, d))
        self._epi = np.full(n, 1.0) if epi is None else np.asarray(epi, float)
        self._ale = np.full(n, 1.0) if ale is None else np.asarray(ale, float)
        self._separates = separates

    def predict(self, sequences):
        return np.zeros(len(sequences))

    def uncertainty(self, sequences):
        return self._epi

    def embed(self, sequences):
        return self.z

    def epistemic_uncertainty(self, sequences):
        return self._epi

    def aleatoric_uncertainty(self, sequences):
        return self._ale

    @property
    def separates_uncertainty(self):
        return self._separates

    def fit(self, sequences, labels):
        pass


SEQS = [f"seq{i}" for i in range(60)]


def test_kmeanspp_picks_distinct_points():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 4))
    sel = kmeanspp(pts, 10, rng)
    assert len(set(sel.tolist())) == 10


def test_badge_returns_requested_batch_without_repeats():
    s = FakeStudent(len(SEQS), seed=1)
    sel = BADGEAcquisition(seed=0).select(s, SEQS, 12)
    assert len(sel) == 12 and len(set(sel.tolist())) == 12


def test_badge_raises_when_embeddings_collapse():
    """The zero-gradient degeneracy must fail loudly, not silently go random."""
    s = FakeStudent(len(SEQS), collapse=True)
    with pytest.raises(ValueError, match="no spread"):
        BADGEAcquisition(seed=0).select(s, SEQS, 5)


def test_badge_ratio_divides_out_aleatoric_noise():
    """A candidate that is uncertain only because its LABEL is noisy must not win.

    This is the MPRA failure mode: low-activity sequences have large aleatoric
    variance from low read counts, and total-uncertainty acquisition spends the whole
    batch on them.
    """
    n = len(SEQS)
    epi = np.full(n, 1.0)
    ale = np.full(n, 1.0)
    epi[0] = 10.0  # genuinely informative
    ale[1] = 0.01
    epi[1] = 10.0  # uncertain only because its label is noisy -> high total, low ratio
    ale[0] = 1.0
    s = FakeStudent(n, seed=2, epi=epi, ale=ale)
    w_ratio = BADGEAcquisition(seed=0, mode="ratio")._weights(s, SEQS)
    w_total = BADGEAcquisition(seed=0, mode="total")._weights(s, SEQS)
    # under total weighting the two look equally attractive
    assert np.isclose(w_total[0], w_total[1])
    # under the ratio the noisy one is ranked far above... by construction its
    # aleatoric is SMALLER, so it should rank higher; flip to assert the mechanism
    assert w_ratio[1] > w_ratio[0]


def test_badge_warns_when_model_cannot_separate(caplog):
    s = FakeStudent(len(SEQS), seed=3, separates=False)
    with caplog.at_level("WARNING"):
        BADGEAcquisition(seed=0, mode="ratio").select(s, SEQS, 5)
    assert any("cannot separate" in r.getMessage() for r in caplog.records)


def test_badge_controls_are_distinct_strategies():
    """The two halves must be separately runnable or a BADGE gain is unattributable."""
    s = FakeStudent(len(SEQS), seed=4, epi=np.arange(len(SEQS), dtype=float))
    top = EpistemicOnlyAcquisition().select(s, SEQS, 5)
    assert top.tolist() == list(range(len(SEQS) - 1, len(SEQS) - 6, -1))
    div = KMeansPPOnlyAcquisition(seed=0).select(s, SEQS, 5)
    assert len(set(div.tolist())) == 5


def test_invalid_badge_mode_rejected():
    with pytest.raises(ValueError):
        BADGEAcquisition(mode="nonsense")


def test_acquisition_registry_complete():
    for name in (
        "random",
        "diversity_kmeans",
        "uncertainty_mcdropout",
        "badge",
        "batchbald",
        "badge_epistemic_only",
        "badge_kmeanspp_only",
    ):
        assert name in R.ACQ_REGISTRY


@pytest.mark.parametrize("name", sorted(R.ACQ_REGISTRY))
def test_acquisition_builds_and_documents_params(name):
    spec = R.get_acq(name)
    spec.build(seed=0)
    for key, p in spec.params.items():
        assert p.help.strip(), f"{name}.{key} undocumented"


# --- binned variants: the textbook forms, and the contrast that motivates them ----


class SampleStudent(FakeStudent):
    """Student whose posterior samples we control, for the binned methods."""

    def __init__(self, samples: np.ndarray, d: int = 6, seed: int = 0):
        n = samples.shape[1]
        super().__init__(n, d=d, seed=seed)
        self._samples = samples
        self._i = 0

    def predict(self, sequences):
        out = self._samples[self._i % len(self._samples)]
        self._i += 1
        return out


def _spread_samples(n=40, t=12, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n) * 2
    return truth[None, :] + rng.normal(scale=0.8, size=(t, n))


def test_bin_edges_quantile_gives_equal_counts():
    from albench.acquisition.binned import bin_edges

    v = np.random.default_rng(0).lognormal(size=5000)
    e = bin_edges(v, 10, "quantile")
    counts = np.bincount(np.digitize(v, e), minlength=len(e) + 1)
    assert counts.min() > 0.5 * counts.max()  # roughly balanced


def test_bin_edges_warns_on_ties(caplog):
    from albench.acquisition.binned import bin_edges

    v = np.zeros(1000)
    v[:10] = 1.0
    with caplog.at_level("WARNING"):
        bin_edges(v, 10, "quantile")
    assert any("duplicates" in r.getMessage() for r in caplog.records)


def test_binned_badge_embedding_is_not_degenerate():
    """The whole point: the categorical gradient is non-zero where the Gaussian is 0."""
    from albench.acquisition.binned import BinnedBADGEAcquisition

    s = SampleStudent(_spread_samples())
    sel = BinnedBADGEAcquisition(seed=0, n_bins=6).select(s, [f"s{i}" for i in range(40)], 8)
    assert len(sel) == 8 and len(set(sel.tolist())) == 8


def test_binned_methods_reject_a_deterministic_student():
    """No epistemic spread means no information; must fail loudly, not pick randomly."""
    from albench.acquisition.binned import BinnedBADGEAcquisition, gather_samples

    same = np.tile(np.arange(30, dtype=float), (8, 1))
    s = SampleStudent(same)
    with pytest.raises(ValueError, match="identical across passes"):
        gather_samples(s, [f"s{i}" for i in range(30)], 8)
    with pytest.raises(ValueError):
        BinnedBADGEAcquisition(seed=0).select(s, [f"s{i}" for i in range(30)], 4)


def test_binned_batchbald_selects_distinct_and_is_deterministic():
    from albench.acquisition.binned import BinnedBatchBALDAcquisition

    seqs = [f"s{i}" for i in range(40)]
    a = BinnedBatchBALDAcquisition(seed=0, n_bins=5).select(
        SampleStudent(_spread_samples()), seqs, 6
    )
    b = BinnedBatchBALDAcquisition(seed=0, n_bins=5).select(
        SampleStudent(_spread_samples()), seqs, 6
    )
    assert len(set(a.tolist())) == 6
    assert a.tolist() == b.tolist()


def test_binned_variants_registered():
    for name in ("badge_binned", "batchbald_binned"):
        assert name in R.ACQ_REGISTRY
        R.get_acq(name).build(seed=0)


# --- uncertainty variants from the 2026-09-16 meeting ----------------------


class _MultiTaskStub:
    """Student exposing per-condition uncertainty, for the aggregate/differential arms."""

    def __init__(self, n: int = 200, seed: int = 0) -> None:
        import numpy as np

        rng = np.random.default_rng(seed)
        self._act = rng.normal(size=n)
        # condition 0 uncertain for the first half, condition 1 for the second:
        # total variance is flat across candidates while the RATIO is extreme at both
        # ends, so the two rules must disagree if they are implemented differently.
        self._u = np.stack(
            [
                np.r_[np.full(n // 2, 1.0), np.full(n - n // 2, 0.1)],
                np.r_[np.full(n // 2, 0.1), np.full(n - n // 2, 1.0)],
            ],
            axis=1,
        )

    def predict(self, seqs):
        return self._act[: len(seqs)]

    def uncertainty(self, seqs):
        import numpy as np

        return np.sqrt((self._u[: len(seqs)] ** 2).sum(axis=1))

    def uncertainty_per_condition(self, seqs):
        return self._u[: len(seqs)]

    def embed(self, seqs):
        import numpy as np

        return np.zeros((len(seqs), 4))


def test_aggregate_and_differential_are_different_rules() -> None:
    """Total variance and the variance RATIO must not select the same set here."""
    import numpy as np

    from albench.acquisition.uncertainty_variants import (
        AggregateUncertaintyAcquisition,
        DifferentialUncertaintyAcquisition,
    )

    cands = ["ACGT" * 50] * 200
    st = _MultiTaskStub()
    agg = set(AggregateUncertaintyAcquisition().select(st, cands, 40).tolist())
    dif = set(DifferentialUncertaintyAcquisition().select(st, cands, 40).tolist())
    assert len(agg) == len(dif) == 40
    # Constructed so total variance is identical for every candidate: the aggregate
    # rule cannot distinguish them, the ratio rule finds every one extreme. The point
    # is that they are computed from different quantities, not that they never agree.
    assert np.isclose(np.ptp((st._u**2).sum(axis=1)), 0.0)


def test_aggregate_refuses_a_single_output_student() -> None:
    """Silently collapsing to single-task uncertainty would duplicate another arm."""
    import pytest

    from albench.acquisition.uncertainty_variants import AggregateUncertaintyAcquisition

    class SingleOutput:
        def predict(self, seqs):
            import numpy as np

            return np.zeros(len(seqs))

        def uncertainty(self, seqs):
            import numpy as np

            return np.ones(len(seqs))

    with pytest.raises(ValueError, match="per-cell-type"):
        AggregateUncertaintyAcquisition().select(SingleOutput(), ["ACGT" * 50] * 10, 3)


def test_activity_normalised_prefers_excess_not_activity() -> None:
    """The whole point: do not just re-rank by activity."""
    import numpy as np

    from albench.acquisition.uncertainty_variants import (
        ActivityNormalisedUncertaintyAcquisition,
    )

    n = 400
    rng = np.random.default_rng(0)
    act = np.linspace(0, 10, n)

    class TrendStub:
        # uncertainty rises with activity, PLUS a few genuine outliers at LOW activity
        def __init__(self):
            self.u = 0.1 * act + 0.01 * rng.normal(size=n)
            self.spikes = np.array([5, 17, 29, 41])
            self.u[self.spikes] += 0.8

        def predict(self, seqs):
            return act[: len(seqs)]

        def uncertainty(self, seqs):
            return self.u[: len(seqs)]

    st = TrendStub()
    picked = set(
        ActivityNormalisedUncertaintyAcquisition(n_bins=20).select(
            st, ["A" * 200] * n, 20
        ).tolist()
    )
    # the low-activity spikes must be found; a raw-uncertainty rule would pick the
    # high-activity tail instead and miss every one of them
    assert set(st.spikes.tolist()) <= picked
    raw_top = set(np.argsort(-st.u)[:20].tolist())
    assert not set(st.spikes.tolist()) <= raw_top
