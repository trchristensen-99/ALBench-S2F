"""Build an AL run from registry names, and expand a parameter screen into jobs.

Two things live here:

``build_run_config``      turns ``(reservoir name + params, acquisition name + params)``
                         into a :class:`~albench.loop.RunConfig` that ALLoop can
                         execute, so a run is specified by configuration rather than
                         by importing and wiring classes by hand.

``expand_screen``         turns one screen config into the full list of cells, which
                         is what makes the parameter screen a single file instead of
                         one file per combination.

THE SCREEN'S SHAPE. Each cell is (strategy, parameter combination, D, seed). D is the
size of the *starting* labelled set, and the two values that matter are 30k and 300k:
the 300k point is the corpus we already have, and 30k is the regime most groups are
actually in. Both have real headroom -- measured 2026-09-14, a student ensemble at
D=300k reaches 0.848 against experimental labels while the oracle reaches 0.916, and
the last doubling still gained +0.056 -- so neither is a saturated dead end.

Replicates are seeds of the RESERVOIR, not just of training: two draws from the same
generative strategy are different sequence sets, and that variation is part of what
the screen is measuring.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

from albench.core.loop import RunConfig
from albench.core.registry import Context, expand_sweep, get, get_acq

logger = logging.getLogger(__name__)


def registry_candidate_provider(
    reservoir: str,
    params: dict[str, Any],
    ctx: Context,
    seed: int = 0,
    fresh_each_round: bool = True,
):
    """A ``(round_idx, n) -> list[str]`` provider backed by a registry strategy.

    ``fresh_each_round`` re-generates candidates every round with a round-dependent
    seed. That is the honest default for generative strategies: re-showing the same
    candidate pool every round would let the acquisition function exhaust it and make
    later rounds look artificially hard.
    """
    spec = get(reservoir)

    def provide(round_idx: int, n: int) -> list[str]:
        s = (seed * 1000 + round_idx) if fresh_each_round else seed
        seqs, _meta = spec.generate(n, ctx, seed=s, **params)
        return list(seqs)

    return provide


def build_run_config(
    *,
    reservoir: str,
    acquisition: str,
    output_dir: str,
    n_rounds: int,
    batch_size: int,
    reservoir_params: dict[str, Any] | None = None,
    acquisition_params: dict[str, Any] | None = None,
    ctx: Context | None = None,
    seed: int = 0,
    n_reservoir_candidates: int = 10_000,
) -> RunConfig:
    """Assemble a RunConfig from registry names, validating parameters up front."""
    r_params = dict(reservoir_params or {})
    a_params = dict(acquisition_params or {})
    r_spec, a_spec = get(reservoir), get_acq(acquisition)

    # Construct both now so a bad parameter fails before any GPU time is spent.
    r_spec.build(seed=seed, **r_params)
    acquirer = a_spec.build(seed=seed, **a_params)

    ctx = ctx or Context()
    provider = registry_candidate_provider(reservoir, r_params, ctx, seed=seed)
    return RunConfig(
        n_rounds=n_rounds,
        batch_size=batch_size,
        reservoir_schedule={"default": None},  # superseded by candidate_provider
        acquisition_schedule={"default": acquirer},
        output_dir=output_dir,
        n_reservoir_candidates=n_reservoir_candidates,
        candidate_provider=provider,
    )


# ---------------------------------------------------------------------------
# Parameter screen
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """One screen cell: a strategy at one parameter setting, size and seed."""

    reservoir: str
    params: dict[str, Any]
    d: int
    seed: int
    acquisition: str = "random"

    @property
    def tag(self) -> str:
        """Filesystem-safe identifier that encodes the whole cell."""
        parts = [self.reservoir, f"d{self.d}", f"seed{self.seed}"]
        if self.acquisition != "random":
            parts.append(f"acq-{self.acquisition}")
        for k, v in sorted(self.params.items()):
            parts.append(f"{k}-{v}")
        return "__".join(str(p) for p in parts).replace(".", "p").replace("/", "-")


@dataclass
class ScreenConfig:
    """A screen: strategies with parameter grids, crossed with sizes and seeds."""

    strategies: dict[str, dict[str, Any]] = field(default_factory=dict)
    d_values: tuple[int, ...] = (30_000, 300_000)
    seeds: tuple[int, ...] = (0, 1, 2)
    acquisitions: tuple[str, ...] = ("random",)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScreenConfig":
        return cls(
            strategies=d.get("strategies", {}),
            d_values=tuple(d.get("d_values", (30_000, 300_000))),
            seeds=tuple(d.get("seeds", (0, 1, 2))),
            acquisitions=tuple(d.get("acquisitions", ("random",))),
        )


def expand_screen(cfg: ScreenConfig, stage: str = "screen") -> list[Cell]:
    """Expand a screen config into cells.

    ``stage="screen"`` varies ONE factor at a time from each strategy's defaults --
    the one-factor-at-a-time design agreed in the meeting, chosen because measured
    replicate spread is ~0.005 in r, so any factor worth a factorial should be
    visible here. ``stage="factorial"`` takes the full cross product instead, for the
    follow-up on whichever factors actually moved.
    """
    if stage not in ("screen", "factorial"):
        raise ValueError(f"stage must be screen/factorial, got {stage!r}")

    cells: list[Cell] = []
    for name, grid in cfg.strategies.items():
        spec = get(name)  # validates the strategy name
        unknown = set(grid) - set(spec.params)
        if unknown:
            raise ValueError(
                f"{name}: screen varies unknown parameter(s) {sorted(unknown)}. "
                f"Tunable: {sorted(spec.params)}"
            )
        if stage == "factorial":
            # Each factor's DEFAULT must be one of its levels, or the grid does not
            # contain the centre point and the factorial cannot be compared against
            # the screen that selected its factors.
            full = {}
            for key, values in grid.items():
                vals = list(values) if isinstance(values, list) else [values]
                default = spec.params[key].default
                if default not in vals:
                    vals.insert(0, default)
                full[key] = vals
            combos = expand_sweep(full) if full else [{}]
        else:
            # centre point plus one arm per (factor, level)
            combos = [{}]
            for key, values in grid.items():
                for v in values if isinstance(values, list) else [values]:
                    if v != spec.params[key].default:
                        combos.append({key: v})
        for combo, d, seed, acq in itertools.product(
            combos, cfg.d_values, cfg.seeds, cfg.acquisitions
        ):
            cells.append(Cell(name, dict(combo), d, seed, acq))
    return cells


def screen_summary(cells: list[Cell], seq_per_s: float = 10_957.0, epochs: int = 60) -> str:
    """Human-readable cost estimate.

    Throughput default is the H100 bf16 figure measured 2026-09-14. This covers
    STUDENT TRAINING only -- oracle labelling is the dominant cost at these sizes and
    is charged once per distinct (strategy, params, D, seed), not once per cell.
    """
    by_strategy: dict[str, int] = {}
    for c in cells:
        by_strategy[c.reservoir] = by_strategy.get(c.reservoir, 0) + 1
    gpu_h = sum(c.d * epochs / seq_per_s / 3600 for c in cells)
    lines = [
        f"{len(cells)} cells across {len(by_strategy)} strategies",
        *(f"    {k:<24} {v:>4} cells" for k, v in sorted(by_strategy.items())),
        f"  student training ~{gpu_h:.1f} GPU-h at {seq_per_s:,.0f} seq/s, {epochs} epochs",
        "  NOTE: oracle labelling is charged separately and usually dominates; "
        "see reference_legnet_throughput.",
    ]
    return "\n".join(lines)
