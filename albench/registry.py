"""One declarative table of reservoir strategies, replacing name-string dispatch.

WHY THIS EXISTS. Strategies used to be selected by name in an if/elif chain that
passed only ``(n, task, seed)``, so constructor parameters were unreachable: varying
a mutation rate meant adding a new YAML file and a new branch (hence
``motif_density_2.yaml`` / ``motif_density_3.yaml``, ``evoaug_heavy``,
``recombination_2pt``). That makes a parameter screen impossible to express and
makes the codebase hard for anyone else to extend.

Here, each strategy declares once:

  factory   how to construct it from parameters
  adapter   how to call its ``generate`` given a Context -- this is where the old
            if/elif chain went, as data rather than control flow
  params    the tunable parameters, with defaults and one-line help

Any parameter can then be overridden or swept from a config or the CLI without
touching code, and ``albench list`` can print what is tunable without anyone
reading the source.

ADDING A STRATEGY: write the sampler, then add one ``register(...)`` call below.
Nothing else in the codebase needs to change.
"""

from __future__ import annotations

import inspect
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Context:
    """Everything a strategy might need besides its own parameters."""

    task: str = "k562"
    pool_sequences: list[str] | None = None
    pool_labels: np.ndarray | None = None
    oracle: Any = None

    def require_pool(self, name: str) -> list[str]:
        if not self.pool_sequences:
            raise ValueError(
                f"Strategy {name!r} needs a genomic pool but none was supplied. "
                f"Pass --pool or set the task's pool loader."
            )
        return self.pool_sequences


@dataclass
class Param:
    """One tunable parameter."""

    default: Any
    help: str
    choices: tuple | None = None

    def coerce(self, value: Any) -> Any:
        """Best-effort cast of a CLI/YAML string to the default's type."""
        if value is None or self.default is None or isinstance(value, type(self.default)):
            return value
        if isinstance(self.default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        for caster in (int, float):
            if isinstance(self.default, caster):
                return caster(value)
        return value


@dataclass
class Spec:
    """A registered strategy."""

    name: str
    factory: Callable[..., Any]
    adapter: Callable[[Any, int, Context], tuple[list[str], Any]]
    params: dict[str, Param] = field(default_factory=dict)
    doc: str = ""
    assets: tuple[str, ...] = ()
    needs_pool: bool = False
    group: str = ""

    def build(self, seed: int | None = None, **overrides) -> Any:
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}. "
                f"Tunable: {sorted(self.params)}"
            )
        kwargs = {k: p.default for k, p in self.params.items()}
        for k, v in overrides.items():
            kwargs[k] = self.params[k].coerce(v)
        for k, v in kwargs.items():
            p = self.params[k]
            if p.choices and v not in p.choices:
                raise ValueError(f"{self.name}.{k}={v!r} not in {list(p.choices)}")
        return self.factory(seed=seed, **kwargs)

    def generate(self, n: int, ctx: Context, seed: int | None = None, **overrides):
        return self.adapter(self.build(seed=seed, **overrides), n, ctx)


REGISTRY: dict[str, Spec] = {}


def register(spec: Spec) -> Spec:
    if spec.name in REGISTRY:
        raise ValueError(f"Strategy {spec.name!r} already registered")
    REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> Spec:
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {sorted(REGISTRY)}\n"
            f"Run `albench list` for parameters."
        )
    return REGISTRY[name]


def expand_sweep(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand any list-valued parameter into the full grid of combinations.

    ``{"mut_rate": [0.01, 0.05], "ti_tv": 2.0}`` -> two dicts. This is what makes a
    parameter screen one config instead of one file per cell. Wrap a value in a
    nested list to pass a literal list as a single value.
    """
    keys = [k for k, v in params.items() if isinstance(v, list)]
    if not keys:
        return [dict(params)]
    grids = [params[k] for k in keys]
    out = []
    for combo in itertools.product(*grids):
        d = dict(params)
        d.update(dict(zip(keys, combo)))
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Strategy table. The seven arms agreed on 2026-09-12, plus the baselines the
# comparison needs. Group names match the meeting's categories.
# ---------------------------------------------------------------------------


def _reg_zoonomia() -> None:
    from albench.reservoir.motif_planted_v2 import PhylogeneticZoonomiaSampler

    register(
        Spec(
            name="zoonomia",
            group="genomic",
            doc=(
                "Zoonomia ortholog CREs. Real per-position substitution rates across "
                "241 mammals, so conserved positions are mutated less than "
                "unconstrained ones. rate_mode='flat' is the control that keeps the "
                "same mutation load but discards the conservation profile."
            ),
            factory=lambda seed=None, **kw: PhylogeneticZoonomiaSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            assets=("zoonomia_rates",),
            params={
                "rate_mode": Param(
                    "per_position_matched",
                    "flat | per_position_matched | per_position",
                    choices=("flat", "per_position_matched", "per_position"),
                ),
                "mut_rate": Param(0.02, "mean substitution rate for flat/matched modes"),
                "ti_tv": Param(2.0, "transition/transversion ratio (mammalian ~2)"),
                "ccre_classes": Param((), "restrict to cCRE classes, e.g. ('pELS','dELS')"),
            },
        )
    )


def _reg_mutagenesis() -> None:
    from albench.reservoir.partial_mutagenesis import PartialMutagenesisSampler

    register(
        Spec(
            name="mutagenesis",
            group="genomic_perturbation",
            doc=(
                "Random mutagenesis of pool sequences at an oracle-tuned rate. "
                "base='zoonomia' mutates Zoonomia ortholog CREs instead of the Gosai "
                "genomic pool, which both grounds the starting sequences "
                "phylogenetically and lifts the capacity ceiling of the Zoonomia arm."
            ),
            factory=lambda seed=None, base="pool", **kw: _MutagenesisWithBase(
                seed=seed, base=base, **kw
            ),
            adapter=lambda s, n, ctx: s.generate(n, ctx),
            needs_pool=True,
            params={
                "base": Param(
                    "pool",
                    "pool (Gosai genomic) | zoonomia (ortholog CREs)",
                    choices=("pool", "zoonomia"),
                ),
                "mutation_rate": Param(0.05, "per-base mutation rate"),
                "mutation_rate_distribution": Param(
                    "fixed", "fixed | uniform | normal", choices=("fixed", "uniform", "normal")
                ),
                "min_rate": Param(0.01, "lower bound when distribution is uniform"),
                "max_rate": Param(0.10, "upper bound when distribution is uniform"),
            },
        )
    )


def _reg_evoaug() -> None:
    from albench.reservoir.evoaug_structural import EvoAugStructuralSampler

    common = {
        "p_deletion": Param(0.3, "probability of a deletion"),
        "p_insertion": Param(0.3, "probability of an insertion"),
        "p_inversion": Param(0.2, "probability of an inversion"),
        "p_translocation": Param(0.15, "probability of a translocation"),
        "p_tandem_dup": Param(0.1, "probability of a tandem duplication"),
        "p_point_mutation": Param(0.3, "probability of point mutation"),
        "point_mutation_rate": Param(0.02, "per-base rate when point mutation fires"),
        "max_indel_size": Param(20, "maximum indel length"),
    }
    register(
        Spec(
            name="evoaug",
            group="genomic_perturbation",
            doc="EvoAug structural perturbations with all settings tunable.",
            factory=lambda seed=None, **kw: EvoAugStructuralSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(
                n, base_sequences=ctx.require_pool("evoaug"), task=ctx.task
            ),
            needs_pool=True,
            params=dict(common),
        )
    )
    register(
        Spec(
            name="evoaug_published",
            group="genomic_perturbation",
            doc=(
                "EvoAug at the settings published in the paper. CONTROL ARM: it exists "
                "so any gain from the tuned arm is attributable to the tuning rather "
                "than to EvoAug itself. Do not sweep this one."
            ),
            factory=lambda seed=None, **kw: EvoAugStructuralSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(
                n, base_sequences=ctx.require_pool("evoaug_published"), task=ctx.task
            ),
            needs_pool=True,
            params={},
        )
    )


def _reg_motif() -> None:
    from albench.reservoir.motif_planted_v2 import MotifPlantedV2Sampler

    shared = {
        "min_motifs": Param(3, "minimum motifs planted per sequence"),
        "max_motifs": Param(7, "maximum motifs planted per sequence"),
        "vocab_cluster_at": Param(0.90, "collapse PFMs above this similarity"),
        "vocab_trim_ic": Param(0.5, "trim terminal positions below this IC (bits)"),
        "vocab_max_len": Param(12, "drop motifs longer than this after trimming"),
        "vocab_size": Param(None, "cap vocabulary size, most-informative first"),
        "plant_mode": Param(
            "pwm_sample",
            "pwm_sample (variable sites) | consensus (memorisation control)",
            choices=("pwm_sample", "consensus"),
        ),
        "cluster_mode": Param(
            "sample_members",
            "sample_members (draw a member PFM per site) | representative",
            choices=("sample_members", "representative"),
        ),
        "motif_mutation_rate": Param(
            0.0,
            "per-base mutation applied to each planted site, to create WEAK binding "
            "sites alongside strong ones (Peter, 2026-09-12)",
        ),
        "ct_bias": Param(
            0.0,
            "fraction of sites drawn from the cell-type-enriched subset; the rest come "
            "from the full vocabulary. 0.2 = upweight K562/HepG2 without restricting.",
        ),
    }
    register(
        Spec(
            name="motif_shared_core",
            group="motif",
            doc=(
                "Motif vocabulary restricted to the shared core: motifs whose binding "
                "preference is common across cell types."
            ),
            factory=lambda seed=None, **kw: MotifPlantedV2Sampler(
                seed=seed, motif_set="jaspar", vocab_subset="shared_core", **kw
            ),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            assets=("bg_cache",),
            params=dict(shared),
        )
    )
    register(
        Spec(
            name="motif_ct_enriched",
            group="motif",
            doc="Motif vocabulary enriched for TFs active in the target cell types.",
            factory=lambda seed=None, **kw: MotifPlantedV2Sampler(
                seed=seed, motif_set="jaspar", vocab_subset="ct_enriched", **kw
            ),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            assets=("bg_cache",),
            params=dict(shared),
        )
    )
    syntax = dict(shared)
    syntax["vocab_size"] = Param(30, "restricted vocabulary: small enough for pair coverage")
    register(
        Spec(
            name="motif_syntax_core",
            group="motif",
            doc=(
                "Syntax core: a RESTRICTED vocabulary crossed combinatorially. Small |V| "
                "is the point -- pairwise coverage is only adequate below ~120 entries, "
                "so grammar/interaction learning needs a narrow vocabulary."
            ),
            factory=lambda seed=None, **kw: MotifPlantedV2Sampler(
                seed=seed, motif_set="jaspar", vocab_subset="syntax_core", **kw
            ),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            assets=("bg_cache",),
            params=syntax,
        )
    )


def _reg_baselines() -> None:
    from albench.reservoir.genomic import GenomicSampler
    from albench.reservoir.random_sampler import RandomSampler

    register(
        Spec(
            name="random",
            group="baseline",
            doc="Uniform random sequences. The floor every strategy must beat.",
            factory=lambda seed=None, **kw: RandomSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            params={},
        )
    )
    register(
        Spec(
            name="genomic",
            group="baseline",
            doc="Real genomic sequences drawn from the existing pool.",
            factory=lambda seed=None, **kw: GenomicSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(
                n, pool_sequences=ctx.require_pool("genomic"), pool_labels=ctx.pool_labels
            ),
            needs_pool=True,
            params={},
        )
    )


class _MutagenesisWithBase:
    """Mutagenesis over either the genomic pool or Zoonomia ortholog CREs.

    Using Zoonomia as the base does two things at once: the starting sequences are
    phylogenetically grounded rather than arbitrary genomic draws, and the arm stops
    being capacity-limited by the 199,373 available Zoonomia regions, since each
    region can yield many distinct mutants.
    """

    def __init__(self, seed=None, base="pool", **kw):
        from albench.reservoir.partial_mutagenesis import PartialMutagenesisSampler

        self.base = base
        self._sampler = PartialMutagenesisSampler(seed=seed, **kw)
        self._rng = np.random.default_rng(seed)

    def generate(self, n: int, ctx: Context):
        if self.base == "zoonomia":
            from albench.paths import resolve

            z = np.load(resolve("zoonomia_rates"), allow_pickle=True)
            base_seqs = [str(s) for s in z["sequences"]]
        else:
            base_seqs = ctx.require_pool("mutagenesis")
        return self._sampler.generate(n, base_sequences=base_seqs, task=ctx.task)


# ---------------------------------------------------------------------------
# Acquisition strategies. MVP per the 2026-09-12 meeting is one simple method per
# family -- k-means for diversity, MC dropout for uncertainty -- plus BADGE and
# BatchBALD, which are uncertainty AND diversity rather than diversity alone, and
# the controls without which a BADGE result cannot be attributed.
# ---------------------------------------------------------------------------

ACQ_REGISTRY: dict[str, Spec] = {}


def register_acq(spec: Spec) -> Spec:
    if spec.name in ACQ_REGISTRY:
        raise ValueError(f"Acquisition {spec.name!r} already registered")
    ACQ_REGISTRY[spec.name] = spec
    return spec


def get_acq(name: str) -> Spec:
    if name not in ACQ_REGISTRY:
        raise KeyError(f"Unknown acquisition {name!r}. Available: {sorted(ACQ_REGISTRY)}")
    return ACQ_REGISTRY[name]


def _reg_acquisition() -> None:
    from albench.acquisition.badge import (
        BADGEAcquisition,
        EpistemicOnlyAcquisition,
        KMeansPPOnlyAcquisition,
    )
    from albench.acquisition.batchbald import BatchBALDAcquisition
    from albench.acquisition.diversity import DiversityAcquisition
    from albench.acquisition.random_acq import RandomAcquisition
    from albench.acquisition.uncertainty import UncertaintyAcquisition

    def _acq(name, factory, doc, group, params=None):
        def make(seed=None, _f=factory, **kw):
            # Not every acquisition class takes a seed (some are deterministic), so
            # only pass it when the constructor actually accepts one.
            sig = inspect.signature(_f)
            if "seed" in sig.parameters:
                kw["seed"] = seed
            return _f(**kw)

        register_acq(
            Spec(
                name=name,
                group=group,
                doc=doc,
                factory=make,
                adapter=lambda s, n, ctx: s,  # acquisition is called via .select()
                params=params or {},
            )
        )

    _acq(
        "random",
        RandomAcquisition,
        "Uniform random selection. The floor every method must beat.",
        "baseline",
    )
    _acq(
        "diversity_kmeans",
        DiversityAcquisition,
        "k-means in embedding space. Diversity only -- the simple baseline the "
        "meeting settled on, and the control for BADGE's diversity half.",
        "diversity",
    )
    _acq(
        "uncertainty_mcdropout",
        UncertaintyAcquisition,
        "Top-k by MC-dropout predictive uncertainty. Uncertainty only.",
        "uncertainty",
    )
    _acq(
        "badge",
        BADGEAcquisition,
        "Uncertainty-weighted embeddings seeded by k-means++, so the batch is jointly "
        "uncertain AND diverse. Weights by epistemic/aleatoric so the budget is not "
        "spent on sequences whose labels are intrinsically unmeasurable.",
        "uncertainty+diversity",
        {
            "mode": Param(
                "ratio", "ratio | epistemic | total", choices=("ratio", "epistemic", "total")
            ),
            "eps": Param(1e-3, "floor on the aleatoric denominator"),
        },
    )
    _acq(
        "batchbald",
        BatchBALDAcquisition,
        "Greedy batch mutual information. Closed form for a Gaussian posterior; warns "
        "when the batch exceeds the posterior rank and picks become arbitrary.",
        "uncertainty+diversity",
        {"n_mc_samples": Param(30, "MC forward passes when the student is not an ensemble")},
    )
    from albench.acquisition.binned import (
        BinnedBADGEAcquisition,
        BinnedBatchBALDAcquisition,
    )

    _binned_params = {
        "n_bins": Param(10, "activity bins; more bins = finer resolution, thinner counts"),
        "bin_mode": Param(
            "quantile",
            "quantile (equal counts) | uniform (equal width)",
            choices=("quantile", "uniform"),
        ),
        "n_mc_samples": Param(30, "MC passes when the student is not an ensemble"),
    }
    _acq(
        "badge_binned",
        BinnedBADGEAcquisition,
        "BADGE in its TEXTBOOK form, on activity discretised into bins. The "
        "cross-entropy gradient (p - onehot) (x) z is non-zero, so the degeneracy that "
        "breaks BADGE on a Gaussian head does not arise. This is why yeast needs no "
        "adaptation -- DREAM activities are already binned -- and it is the arm to "
        "compare against the continuous adaptation for human MPRA.",
        "uncertainty+diversity",
        dict(_binned_params),
    )
    _acq(
        "batchbald_binned",
        BinnedBatchBALDAcquisition,
        "BatchBALD in its TEXTBOOK discrete form on binned activity. Rank is governed "
        "by the bins and the batch rather than by the ensemble size, so it does not "
        "saturate after M-1 picks the way the Gaussian version does.",
        "uncertainty+diversity",
        {**_binned_params, "n_joint_samples": Param(2000, "samples for the joint term")},
    )
    _acq(
        "badge_epistemic_only",
        EpistemicOnlyAcquisition,
        "CONTROL for BADGE: uncertainty without diversity.",
        "control",
    )
    _acq(
        "badge_kmeanspp_only",
        KMeansPPOnlyAcquisition,
        "CONTROL for BADGE: diversity without uncertainty.",
        "control",
    )


def _bootstrap() -> None:
    """Register everything. Import failures name the strategy that could not load."""
    for fn in (
        _reg_zoonomia,
        _reg_mutagenesis,
        _reg_evoaug,
        _reg_motif,
        _reg_baselines,
        _reg_acquisition,
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - a broken strategy must not hide the rest
            import logging

            logging.getLogger(__name__).warning(
                "could not register %s: %s: %s", fn.__name__, type(e).__name__, e
            )


_bootstrap()
