"""The registry contract: every strategy is constructible and every knob is declared.

These run without any data assets, so a fresh clone can verify its install before
downloading anything.
"""

from __future__ import annotations

import pytest

from albench import registry as R


def test_expected_strategies_registered():
    for name in (
        "zoonomia",
        "mutagenesis",
        "evoaug",
        "evoaug_ours_default",
        "motif_shared_core",
        "motif_ct_enriched",
        "motif_syntax_core",
    ):
        assert name in R.REGISTRY, f"{name} failed to register"


@pytest.mark.parametrize("name", sorted(R.REGISTRY))
def test_every_strategy_builds_with_defaults(name):
    """Defaults must construct. Data loading is lazy, so this needs no assets."""
    R.get(name).build(seed=0)


@pytest.mark.parametrize("name", sorted(R.REGISTRY))
def test_params_are_documented(name):
    for key, p in R.get(name).params.items():
        assert p.help.strip(), f"{name}.{key} has no help string"


@pytest.mark.parametrize("name", sorted(R.REGISTRY))
def test_unknown_parameter_is_rejected(name):
    """A typo must fail loudly rather than being silently ignored."""
    with pytest.raises(ValueError, match="unknown parameter"):
        R.get(name).build(seed=0, definitely_not_a_real_param=1)


def test_choices_are_enforced():
    with pytest.raises(ValueError):
        R.get("zoonomia").build(seed=0, rate_mode="nonsense")


def test_sweep_expansion():
    assert len(R.expand_sweep({"a": [1, 2, 3], "b": [10, 20]})) == 6
    assert R.expand_sweep({"a": 1}) == [{"a": 1}]


def test_sweep_is_deterministic_order():
    g = R.expand_sweep({"a": [1, 2], "b": ["x", "y"]})
    assert g == [{"a": 1, "b": "x"}, {"a": 1, "b": "y"}, {"a": 2, "b": "x"}, {"a": 2, "b": "y"}]


def test_context_requires_pool_with_actionable_message():
    ctx = R.Context(task="k562")
    with pytest.raises(ValueError, match="needs a genomic pool"):
        ctx.require_pool("genomic")


def test_wrapper_factories_are_not_handed_base_sequences() -> None:
    """A factory returning a *WithBase wrapper must pair with a ctx-style adapter.

    The wrappers (_EvoAugWithBase, _MutagenesisWithBase) resolve their own starting
    sequences from the `base` parameter, so their generate() takes (n, ctx). The raw
    samplers take (n, base_sequences=..., task=...). Changing a factory from one to the
    other without changing its adapter type-checks fine and only fails at run time:
    it cost an overnight chain, where the generate stage died with
    "_EvoAugWithBase.generate() got an unexpected keyword argument 'base_sequences'"
    and every dependent stage went DependencyNeverSatisfied.
    """
    import inspect

    from albench.core import registry as reg

    for name, spec in reg.REGISTRY.items():
        fsrc = inspect.getsource(spec.factory)
        asrc = inspect.getsource(spec.adapter)
        wrapper = "WithBase(" in fsrc
        passes_base = "base_sequences=" in asrc
        assert not (wrapper and passes_base), (
            f"{name}: factory builds a *WithBase wrapper (which resolves its own base) "
            f"but the adapter still passes base_sequences="
        )
