"""Screen expansion and registry-driven ALLoop wiring."""

from __future__ import annotations

import numpy as np
import pytest

from albench.loop import ALLoop, RunConfig
from albench.model import SequenceModel
from albench.registry import Context
from albench.run import Cell, ScreenConfig, expand_screen, registry_candidate_provider
from albench.task import TaskConfig


def _cfg(**kw):
    base = dict(
        strategies={"random": {}, "zoonomia": {"ti_tv": [1.0, 4.0]}},
        d_values=(30_000, 300_000),
        seeds=(0, 1),
        acquisitions=("random",),
    )
    base.update(kw)
    return ScreenConfig.from_dict(base)


def test_screen_is_one_factor_at_a_time():
    """Centre point plus one arm per (factor, non-default level)."""
    cells = expand_screen(_cfg(), "screen")
    zoo = [c for c in cells if c.reservoir == "zoonomia"]
    combos = {tuple(sorted(c.params.items())) for c in zoo}
    assert () in combos  # the centre point is present
    assert (("ti_tv", 1.0),) in combos
    assert (("ti_tv", 4.0),) in combos
    assert len(combos) == 3


def test_factorial_contains_the_centre_point():
    """Without the default as a level the factorial cannot be compared to the screen."""
    cells = expand_screen(_cfg(), "factorial")
    zoo = [c for c in cells if c.reservoir == "zoonomia"]
    combos = {tuple(sorted(c.params.items())) for c in zoo}
    assert (("ti_tv", 2.0),) in combos  # 2.0 is the registry default


def test_factorial_is_larger_than_the_screen_when_factors_interact():
    """With ONE factor the two designs coincide; the factorial only costs more once
    there is more than one factor to cross."""
    one = _cfg(strategies={"zoonomia": {"ti_tv": [1.0, 4.0]}})
    assert len(expand_screen(one, "factorial")) == len(expand_screen(one, "screen"))

    two = _cfg(strategies={"zoonomia": {"ti_tv": [1.0, 4.0], "mut_rate": [0.01, 0.05]}})
    assert len(expand_screen(two, "factorial")) > len(expand_screen(two, "screen"))


def test_screen_rejects_an_unknown_parameter():
    with pytest.raises(ValueError, match="unknown parameter"):
        expand_screen(_cfg(strategies={"zoonomia": {"not_a_param": [1]}}), "screen")


def test_screen_rejects_an_unknown_strategy():
    with pytest.raises(KeyError):
        expand_screen(_cfg(strategies={"nope": {}}), "screen")


def test_cells_cross_sizes_and_seeds():
    cells = expand_screen(_cfg(), "screen")
    assert {c.d for c in cells} == {30_000, 300_000}
    assert {c.seed for c in cells} == {0, 1}


def test_tags_are_unique_and_filesystem_safe():
    cells = expand_screen(_cfg(), "screen")
    tags = [c.tag for c in cells]
    assert len(set(tags)) == len(tags)
    assert all(not set(t) & set("/. ") for t in tags)


def test_tag_encodes_params():
    t = Cell("zoonomia", {"ti_tv": 4.0}, 30_000, 1).tag
    assert "zoonomia" in t and "d30000" in t and "seed1" in t and "ti_tv" in t


def test_candidate_provider_varies_by_round():
    """Re-showing one pool every round would let acquisition exhaust it."""
    provide = registry_candidate_provider("random", {}, Context(task="k562"), seed=3)
    a, b = provide(0, 50), provide(1, 50)
    assert len(a) == len(b) == 50
    assert a != b


# --- end-to-end: a registry strategy driving ALLoop ------------------------------


class TinyOracle(SequenceModel):
    def predict(self, sequences):
        return np.array([float(s.count("G") + s.count("C")) / max(len(s), 1) for s in sequences])

    def uncertainty(self, sequences):
        return np.ones(len(sequences))

    def embed(self, sequences):
        return np.zeros((len(sequences), 2))

    def fit(self, sequences, labels):
        pass


class TinyStudent(TinyOracle):
    def __init__(self):
        self.n_fit = 0
        self._rng = np.random.default_rng(0)

    def embed(self, sequences):
        return self._rng.normal(size=(len(sequences), 4))

    def uncertainty(self, sequences):
        return self._rng.random(len(sequences)) + 0.1

    def epistemic_uncertainty(self, sequences):
        return self._rng.random(len(sequences)) + 0.1

    def fit(self, sequences, labels):
        self.n_fit += 1


def test_registry_strategy_drives_alloop_end_to_end(tmp_path):
    from albench.registry import get_acq

    provide = registry_candidate_provider("random", {}, Context(task="k562"), seed=1)
    cfg = RunConfig(
        n_rounds=2,
        batch_size=8,
        reservoir_schedule={"default": None},
        acquisition_schedule={"default": get_acq("badge").build(seed=0)},
        output_dir=str(tmp_path),
        n_reservoir_candidates=40,
        candidate_provider=provide,
    )
    loop = ALLoop(
        task=TaskConfig(
            name="k562", organism="human", sequence_length=200, data_root=str(tmp_path)
        ),
        oracle=TinyOracle(),
        student=TinyStudent(),
        initial_labeled=["ACGT" * 50, "TGCA" * 50],
        run_config=cfg,
    )
    r = loop.step()
    assert r is not None
    assert len(r.selected_sequences) == 8
    assert loop.n_labeled == 10  # 2 seed + 8 selected
