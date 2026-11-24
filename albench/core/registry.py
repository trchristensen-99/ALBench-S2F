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


def _reg_encode_accessibility() -> None:
    from albench.reservoir.encode_accessibility import EncodeAccessibilitySampler

    register(
        Spec(
            name="encode_accessibility",
            group="descoped",
            doc=(
                "DESCOPED at the 2026-09-09 meeting: ENCODE regions are largely "
                "already covered by the Gosai library and the remainder adds only "
                "~100k, so the arm cannot scale. Kept registered so the decision is "
                "visible and reversible, but excluded from the strategy tables. "
                "Fixed-length windows centred on ENCODE DNase/ATAC peaks. Unlike the "
                "Gosai genomic pool, which is whatever the original library happened "
                "to tile, this samples directly from measured accessibility, so the "
                "partition chooses the cell-type contrast: shared_open_both is "
                "accessible in K562 AND HepG2, while k562_only/hepg2_only isolate "
                "cell-type-specific regulatory sequence."
            ),
            factory=lambda seed=None, **kw: EncodeAccessibilitySampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            assets=("encode_peaks", "hg38"),
            params={
                "partition": Param(
                    "shared_open_both",
                    "which peak set to draw from",
                    choices=(
                        "shared_open_both",
                        "k562_only",
                        "hepg2_only",
                        "k562_all",
                        "hepg2_all",
                    ),
                ),
                "seq_len": Param(200, "output window length in bp"),
                "min_peak_width": Param(
                    0,
                    "drop peaks narrower than this; k562_only peaks are often 64bp "
                    "slivers, so raise it to avoid windows that are mostly flank",
                ),
                "primary_chroms_only": Param(True, "drop scaffolds and alt contigs"),
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
        "base": Param(
            "pool",
            "pool (Gosai genomic) | zoonomia (ortholog CREs, scales past the CRE ceiling)",
            choices=("pool", "zoonomia"),
        ),
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
            doc=(
                "EvoAug structural perturbations with all settings tunable. base="
                "'zoonomia' perturbs ortholog CREs instead of the Gosai pool, which "
                "grounds the starting sequences phylogenetically and lifts the capacity "
                "ceiling that caps any resample-only arm at 314,981 real CREs."
            ),
            factory=lambda seed=None, base="pool", **kw: _EvoAugWithBase(
                seed=seed, base=base, **kw
            ),
            # _EvoAugWithBase.generate takes ctx and resolves its own base sequences,
            # so it must NOT be handed base_sequences the way the fixed controls are.
            adapter=lambda s, n, ctx: s.generate(n, ctx),
            needs_pool=True,
            params=dict(common),
        )
    )
    register(
        Spec(
            name="evoaug_ours_default",
            group="genomic_perturbation",
            doc=(
                "Our EvoAugStructuralSampler at ITS OWN defaults, unswept. Formerly "
                "and wrongly named 'evoaug_published': it never carried the published "
                "values, it just declined to override ours, and the parameterisation "
                "differs from the paper's anyway. Kept as the untuned-baseline arm. "
                "For the real published configuration use 'evoaug_paper2023'."
            ),
            factory=lambda seed=None, **kw: EvoAugStructuralSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(
                n, base_sequences=ctx.require_pool("evoaug_ours_default"), task=ctx.task
            ),
            needs_pool=True,
            params={},
        )
    )

    register(
        Spec(
            name="evoaug_paper2023",
            group="genomic_perturbation",
            doc=(
                "EvoAug at the values published in Lee, Yu & Koo 2023, read from the "
                "reference implementation (p-koo/evoaug): deletion/insertion/inversion "
                "and translocation shift all drawn uniformly up to 20bp, mutation "
                "fraction 0.05, and exactly max_augs_per_seq augmentations per "
                "sequence (hard_aug=True). This is the CONTROL ARM: a gain in the "
                "swept 'evoaug' arm is only attributable to tuning if measured against "
                "the published settings, which 'evoaug_ours_default' never was. "
                "KNOWN DEVIATIONS, both forced by our sampler operating on discrete "
                "sequences rather than one-hot tensors during training: there is no "
                "reverse-complement op and no Gaussian-noise op, and point mutation is "
                "applied independently rather than drawn from the same uniform op list. "
                "Do not sweep this one."
            ),
            factory=lambda seed=None, **kw: EvoAugStructuralSampler(
                seed=seed,
                # Paper magnitudes. Insertion and deletion share max_indel_size, which
                # is correct here only because the paper sets both maxima to 20.
                max_indel_size=20,
                max_inversion_size=20,
                max_translocation_size=20,
                point_mutation_rate=0.05,
                # hard_aug=True means a FIXED count per sequence, so min == max.
                min_events=2,
                max_events=2,
                # Equal weight over the paper's structural ops; tandem duplication is
                # ours, not theirs, so it is switched off rather than left at 0.1.
                p_deletion=0.25,
                p_insertion=0.25,
                p_inversion=0.25,
                p_translocation=0.25,
                p_tandem_dup=0.0,
                p_point_mutation=0.4,
                **kw,
            ),
            adapter=lambda s, n, ctx: s.generate(
                n, base_sequences=ctx.require_pool("evoaug_paper2023"), task=ctx.task
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
                seed=seed, motif_set="pfm", vocab_subset="shared_core", **kw
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
                seed=seed, motif_set="pfm", vocab_subset="ct_enriched", **kw
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
                seed=seed, motif_set="pfm", vocab_subset="syntax_core", **kw
            ),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            assets=("bg_cache",),
            params=syntax,
        )
    )

    # The DISCRIMINATION arm. Every motif reservoir above restricts the vocabulary to one
    # annotation class, so a student trained on it has never seen a motif that is present
    # and inactive -- every motif in its experience predicted activity, and it cannot have
    # learned to tell the three classes apart. This one plants from all of them at once.
    #
    # Two of the tiers test the annotation rather than trusting it. `hepg2_only` is
    # annotated active in HepG2 and NOT in K562, so under the annotation it should be
    # inert here; `other` is in the vocabulary and in none of the annotated sets at all.
    # ChIP-derived activity calls are noisy and incomplete, so "inactive here" is a
    # hypothesis -- and a sequence whose planted site the annotation calls dead but the
    # oracle scores as live is exactly the observation that corrects the annotation. A
    # library that only ever plants motifs already believed active cannot make it.
    tiered = dict(shared)
    tiered["tier_weights"] = Param(
        "shared_core=0.35,k562_only=0.30,hepg2_only=0.20,decoy=0.15",
        "fraction of planted SITES per annotation tier: shared_core | k562_only | "
        "hepg2_only | ct_enriched | k562 | hepg2 | other | decoy (dinucleotide-shuffled "
        "motif: same length and composition, no binding preference)",
    )
    tiered.pop("ct_bias")  # mutually exclusive: both reweight the same per-site draw
    # Without this the tier arm is not density-comparable to the arms it is read against.
    # Measured over 200 sequences: the unrestricted 628-entry vocabulary finds 179 native
    # "matches" per 200-bp window and plants 2.21 of the 3-7 sites asked for, against 5.10
    # for motif_shared_core and 4.61 for motif_ct_enriched. A 200-bp window does not hold
    # 179 binding sites -- consensus strings run as short as 4 bp and match by chance --
    # so the blocked positions are arithmetic. 8 bp keeps the protection where a match
    # means something. The restriction is set HERE rather than in the sampler default so
    # that no arm measured before 2026-11-24 changes behaviour.
    tiered["native_min_len"] = Param(
        8, "ignore native matches shorter than this when reserving positions"
    )
    tiered["tier_scope"] = Param(
        "site",
        "site (every sequence is a blend in the requested proportions) | sequence (each "
        "sequence commits to ONE tier, so the weights are the fraction of PURE "
        "cell-type-specific / shared / inactive sequences, with 'mixed' for blended ones)",
        choices=("site", "sequence"),
    )
    register(
        Spec(
            name="motif_tiered",
            group="motif",
            doc=(
                "Motifs planted from SEVERAL annotation tiers at once -- shared, "
                "cell-type-specific, annotated-inactive-here, and non-functional decoys -- "
                "so a student can learn to tell them apart instead of only ever seeing "
                "motifs that work. Also the arm that tests the activity annotation itself."
            ),
            factory=lambda seed=None, **kw: MotifPlantedV2Sampler(seed=seed, motif_set="pfm", **kw),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            assets=("bg_cache",),
            params=tiered,
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


def _resolve_base_sequences(base: str, ctx: "Context", who: str) -> list[str]:
    """Starting sequences a derived strategy perturbs.

    'pool' is the Gosai genomic library. 'zoonomia' is the ortholog CRE set, which is
    both phylogenetically grounded and the answer to a capacity problem: the real-CRE
    pool holds 314,981 distinct sequences, so any arm that merely RESAMPLES it is
    capped there, while an arm that PERTURBS a base can produce many distinct
    descendants per starting sequence and keep scaling past that ceiling.
    """
    if base == "zoonomia":
        from albench.paths import resolve

        z = np.load(resolve("zoonomia_rates"), allow_pickle=True)
        return [str(x) for x in z["sequences"]]
    if base != "pool":
        raise ValueError(f"{who}: unknown base {base!r}; expected 'pool' or 'zoonomia'")
    return ctx.require_pool(who)


class _EvoAugWithBase:
    """EvoAug structural perturbations over the genomic pool or Zoonomia CREs.

    Same motivation as the mutagenesis variant: perturbing a phylogenetically grounded
    base lifts the capacity ceiling that constrains any resample-only arm, and lets the
    EvoAug curve be extended past the point where real CREs run out.
    """

    def __init__(self, seed=None, base="pool", **kw):
        from albench.reservoir.evoaug_structural import EvoAugStructuralSampler

        self.base = base
        self._sampler = EvoAugStructuralSampler(seed=seed, **kw)

    def generate(self, n: int, ctx: "Context"):
        base_seqs = _resolve_base_sequences(self.base, ctx, "evoaug")
        return self._sampler.generate(n, base_sequences=base_seqs, task=ctx.task)


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
        base_seqs = _resolve_base_sequences(self.base, ctx, "mutagenesis")
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


def _reg_sheet_additions() -> None:
    """Strategies from the sequence-budget sheet that had code but no registry entry.

    An unregistered sampler is unusable: `albench generate` dispatches by name, so a
    module sitting in albench/reservoir/ with no Spec cannot be swept, cached or put
    on a curve. These were in that state.
    """
    from albench.reservoir.gc_matched import GCMatchedSampler
    from albench.reservoir.in_silico_evolution_generative import (
        InSilicoEvolutionGenerativeSampler,
    )
    from albench.reservoir.random_sampler import RandomSampler

    register(
        Spec(
            name="dinuc_shuffle",
            group="baseline",
            doc=(
                "Dinucleotide-preserving shuffles of real CREs. Destroys motif syntax "
                "while holding dinucleotide composition fixed, so it isolates how much "
                "of the genomic arm's value is composition rather than arrangement."
            ),
            factory=lambda seed=None, **kw: RandomSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(
                n,
                task=ctx.task,
                method="dinuc_shuffle",
                reference_sequences=ctx.require_pool("dinuc_shuffle"),
            ),
            needs_pool=True,
            params={},
        )
    )

    register(
        Spec(
            name="gc_matched",
            group="baseline",
            doc=(
                "Random sequence with the GC distribution of the real CRE pool. The "
                "control for 'is the genomic arm just GC content?' -- it matches "
                "composition and nothing else."
            ),
            factory=lambda seed=None, **kw: GCMatchedSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(
                n, pool_sequences=ctx.require_pool("gc_matched"), task=ctx.task
            ),
            needs_pool=True,
            params={"n_gc_bins": Param(50, "histogram bins used to match the GC distribution")},
        )
    )

    from albench.reservoir.gc_random import GCRandomSampler

    register(
        Spec(
            name="random_gc65",
            group="baseline",
            doc=(
                "Uniform-random DNA at a FIXED high GC (0.65). Probes the high-GC arm of the "
                "random activity response directly, with no pool and no dinucleotide/motif "
                "structure -- the composition control for 'does high GC alone drive activity?'."
            ),
            factory=lambda seed=None, **kw: GCRandomSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            params={"target_gc": Param(0.65, "fixed GC fraction for every sequence")},
        )
    )
    register(
        Spec(
            name="random_gc_sweep",
            group="baseline",
            doc=(
                "Uniform-random DNA spread EVENLY across GC levels {0.25..0.75}, each level "
                "synthesised as its own batch then pooled. One reservoir spanning a GC range "
                "instead of committing to a single GC; cost is linear in the number of levels."
            ),
            factory=lambda seed=None, **kw: GCRandomSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            params={
                "gc_levels": Param(
                    "0.25,0.35,0.45,0.55,0.65,0.75",
                    "comma-separated GC levels sampled in equal proportion",
                ),
                "target_gc": Param(0.5, "unused when gc_levels is set"),
            },
        )
    )

    def _ise(seed=None, **kw):
        return InSilicoEvolutionGenerativeSampler(seed=seed, **kw)

    def _ise_adapter_named(name: str = "ise_maximize"):
        """An ISE adapter that knows which arm it is.

        The name only affects error messages, but it affects the two that matter: a
        missing pool and a missing fitness model both have to say which of the five
        objectives was being built, or the failure is unattributable in an array job.
        """

        def adapt(s, n, ctx):
            # REFUSE to run without a fitness model. The sampler's own fallback is
            # random mutagenesis at 5%, which would make this arm a silent duplicate of
            # the mutagenesis arm under a different name -- exactly the failure that
            # made the tuned and untuned EvoAug arms byte-identical.
            model = getattr(ctx, "oracle", None)
            if model is None:
                raise ValueError(
                    f"{name} is model-in-the-loop: it needs a predictor to score each "
                    "generation. Pass an oracle in the Context. Running it without one "
                    "silently degrades to random mutagenesis, which would duplicate the "
                    "mutagenesis arm rather than evolve anything."
                )
            return s.generate(
                n,
                base_sequences=ctx.require_pool(name),
                task=ctx.task,
                student_model=model,
            )

        return adapt

    _ise_adapter = _ise_adapter_named("ise_maximize")

    register(
        Spec(
            name="ise_maximize",
            group="model_generative",
            doc=(
                "In-silico evolution toward HIGH activity (sheet row E1, 'BOTH-high'). "
                "Model-in-the-loop: each generation is scored by a predictor and the "
                "elite fraction is carried forward. Needs an oracle in the Context. "
                "NOTE: this is single-objective; the DIFFERENTIAL variant (E2) needs a "
                "multi-cell-type predictor the current sampler does not support."
            ),
            factory=_ise,
            adapter=_ise_adapter,
            needs_pool=True,
            assets=("oracle",),
            params={
                "n_evolution_rounds": Param(5, "generations of mutate-score-select"),
                "mutation_rate": Param(0.05, "per-base rate within a generation"),
                "population_size": Param(100, "candidates per generation"),
                "elite_fraction": Param(0.2, "top fraction carried forward"),
                "evolution_mode": Param(
                    "maximize", "maximize | target", choices=("maximize", "target")
                ),
                "temperature": Param(
                    1.0,
                    "softmax temperature on the selection step; higher keeps more of "
                    "the population, so the arm explores instead of converging",
                ),
            },
        )
    )

    # WHICH PART OF THE ACTIVITY RANGE TO SPEND OLIGOS ON. `ise_maximize` answers only
    # "make it as active as possible", which is one corner of a design space the
    # sampler already implements and the registry never exposed: minimize, hit a
    # target, or spread targets across the range. For a library this is the live
    # question -- a training corpus concentrated at the top of the dynamic range
    # teaches a student nothing about the rest of it -- so each objective is its own
    # registered arm with a fixed activity target, rather than a tuple parameter that
    # would have to survive string coercion through --set.
    #
    # Read every one of these against `mutagenesis` at the same per-base rate: the
    # evolution loop costs oracle calls that random mutagenesis does not, so the arm
    # only earns its place where it beats mutagenesis, not where it beats random.
    _ISE_VARIANTS = [
        (
            "ise_minimize",
            {"evolution_mode": "minimize"},
            "Evolution toward LOW activity: silencers and dead sequence. The negative "
            "half of the dynamic range is where every student is worst and where the "
            "genomic pool has least to say, since real CREs are selected for function.",
        ),
        (
            "ise_target_mid",
            {"evolution_mode": "target", "target_activity": 0.5},
            "Evolution toward MID activity. Deliberately the hardest target: the "
            "measurement ceiling in the middle tertile is ~0 (Nov 20), so this arm "
            "doubles as a test of whether a reservoir can be aimed at a region the "
            "assay cannot resolve.",
        ),
        (
            "ise_diverse",
            {"evolution_mode": "diverse_targets", "activity_distribution": "uniform"},
            "Evolution toward targets drawn UNIFORMLY across the activity range -- one "
            "lineage aimed high, the next low. This is the arm that corresponds to the "
            "library-design instinct of covering the dynamic range rather than its top, "
            "and the one to beat if 'spread the budget' is to be the recommendation.",
        ),
        (
            "ise_explore",
            {"evolution_mode": "maximize", "temperature": 4.0},
            "The same maximisation objective as ise_maximize at a high selection "
            "temperature, so the elite set stays diverse. Isolates how much of any "
            "ise_maximize result is the objective and how much is the loss of "
            "diversity that greedy selection causes.",
        ),
    ]
    for _name, _fixed, _doc in _ISE_VARIANTS:
        register(
            Spec(
                name=_name,
                group="model_generative",
                doc=_doc + " Model-in-the-loop: needs a predictor in the Context.",
                factory=lambda seed=None, _f=_fixed, **kw: _ise(seed=seed, **{**_f, **kw}),
                adapter=_ise_adapter_named(_name),
                needs_pool=True,
                assets=("oracle",),
                params={
                    "n_evolution_rounds": Param(5, "generations of mutate-score-select"),
                    "mutation_rate": Param(0.05, "per-base rate within a generation"),
                    "population_size": Param(100, "candidates per generation"),
                    "elite_fraction": Param(0.2, "top fraction carried forward"),
                    "temperature": Param(1.0, "softmax temperature on selection"),
                },
            )
        )

    # FastSeqProp -- gradient design, not evolution. Gosai et al. built their designed
    # high-activity set with this (plus simulated annealing and AdaLead), and that set is
    # one of our eval corpora, so it is worth having a reservoir that reaches sequence
    # space the way the thing we are evaluated on was reached. Every ise_ arm explores by
    # mutation and therefore stays near sequences it has already scored; this optimises
    # the input directly and goes somewhere else.
    from albench.reservoir.fastseqprop import FastSeqPropSampler

    _FSP_VARIANTS = [
        ("fastseqprop", {"objective": "maximize"}, "Gradient ascent toward HIGH activity."),
        ("fastseqprop_min", {"objective": "minimize"}, "Gradient descent toward LOW activity."),
        (
            "fastseqprop_target",
            {"objective": "target", "target_activity": 0.5},
            "Gradient design toward a MID target, so the corpus covers the middle of the "
            "range rather than only its ends.",
        ),
        (
            "fastseqprop_random_init",
            {"objective": "maximize", "init": "random"},
            "Gradient ascent from NOISE rather than from a real CRE. The paired control "
            "for how much of the designed sequence is the starting CRE and how much the "
            "optimisation actually put there.",
        ),
    ]
    for _name, _fixed, _doc in _FSP_VARIANTS:
        register(
            Spec(
                name=_name,
                group="model_generative",
                doc=(
                    _doc + " Needs a DIFFERENTIABLE guide (a LegNet bank, not the "
                    "AlphaGenome oracle): it takes gradients through the input one-hot."
                ),
                factory=lambda seed=None, _f=_fixed, **kw: FastSeqPropSampler(
                    seed=seed, **{**_f, **kw}
                ),
                adapter=_ise_adapter_named(_name),
                needs_pool=True,
                assets=("oracle",),
                params={
                    "n_steps": Param(150, "Adam steps per batch; cost is linear in this"),
                    "lr": Param(0.1, "Adam learning rate on the logits"),
                    "batch_size": Param(512, "designs optimised simultaneously"),
                    "guide_members": Param(2, "bank members to backprop through (None=all)"),
                    "init": Param(
                        "pool",
                        "pool (start from a real CRE) | random (start from noise)",
                        choices=("pool", "random"),
                    ),
                    "tau": Param(1.0, "softmax temperature"),
                    "entropy_weight": Param(0.0, "penalty keeping the distribution open"),
                },
            )
        )

    # The other two methods that built the Gosai designed set. They reach different
    # sequences and fail differently from a gradient method, so a corpus that wants to
    # cover the designed regime needs all three rather than one corner of it.
    from albench.reservoir.design_methods import AdaLeadSampler, SimulatedAnnealingSampler

    _DESIGN_VARIANTS = [
        (
            "sim_anneal",
            SimulatedAnnealingSampler,
            {"objective": "maximize"},
            "Metropolis walk over single-base edits at a cooling temperature.",
            {
                "n_steps": Param(300, "Metropolis proposals per design"),
                "t0": Param(1.0, "starting temperature, in oracle label units"),
                "t_end": Param(0.01, "final temperature (geometric schedule)"),
                "n_mut": Param(1, "bases changed per proposal"),
            },
        ),
        (
            "sim_anneal_target",
            SimulatedAnnealingSampler,
            {"objective": "target", "target_activity": 0.5},
            "Annealing toward a MID target rather than the top of the range.",
            {
                "n_steps": Param(300, "Metropolis proposals per design"),
                "t0": Param(1.0, "starting temperature, in oracle label units"),
                "t_end": Param(0.01, "final temperature (geometric schedule)"),
                "n_mut": Param(1, "bases changed per proposal"),
            },
        ),
        (
            "adalead",
            AdaLeadSampler,
            {"objective": "maximize"},
            "Threshold-greedy evolutionary search with rollout (Sinai et al. 2020).",
            {
                "n_rounds": Param(10, "evolutionary rounds"),
                "population": Param(16, "children proposed per design per round"),
                "kappa": Param(0.05, "parent threshold; smaller is greedier"),
                "mutation_rate": Param(0.02, "per-base rate when proposing a child"),
                "rollout": Param(4, "extra mutate-and-keep steps on an improving child"),
            },
        ),
        (
            "adalead_target",
            AdaLeadSampler,
            {"objective": "target", "target_activity": 0.5},
            "AdaLead toward a MID target.",
            {
                "n_rounds": Param(10, "evolutionary rounds"),
                "population": Param(16, "children proposed per design per round"),
                "kappa": Param(0.05, "parent threshold; smaller is greedier"),
                "mutation_rate": Param(0.02, "per-base rate when proposing a child"),
                "rollout": Param(4, "extra mutate-and-keep steps on an improving child"),
            },
        ),
    ]
    for _name, _cls, _fixed, _doc, _params in _DESIGN_VARIANTS:
        register(
            Spec(
                name=_name,
                group="model_generative",
                doc=(
                    _doc + " Model-in-the-loop: needs a predictor in the Context. Unlike "
                    "FastSeqProp it needs only predict(), so the oracle COULD guide it -- "
                    "it should not, for the reason in albench/ensemble/bank.py."
                ),
                factory=lambda seed=None, _c=_cls, _f=_fixed, **kw: _c(seed=seed, **{**_f, **kw}),
                adapter=_ise_adapter_named(_name),
                needs_pool=True,
                assets=("oracle",),
                params={**_params, "batch_size": Param(4096, "sequences per predict() call")},
            )
        )

    from albench.reservoir.zoonomia_orthologs import ZoonomiaOrthologSampler

    register(
        Spec(
            name="zoonomia_orthologs",
            group="genomic",
            doc=(
                "REAL orthologous CRE sequences from the 241-mammal alignment -- for "
                "each human cCRE window, the aligned sequence in other mammals. "
                "Replaces the earlier `zoonomia` arm, which mutated HUMAN sequence "
                "under conservation-derived rates and was therefore capped at the "
                "199,373 human cCREs that exist. Measured capacity is ~8.2M sequences "
                "(199,373 windows x the ~41 species callable at min_called_frac=0.95; "
                "the naive 240-species figure overstates it), which still makes it "
                "the most scalable genomic arm rather than the most limited. `distance` selects an evolutionary tier by "
                "observed identity to human: near relatives are nearly human and "
                "test little, distant ones approach novel sequence."
            ),
            factory=lambda seed=None, **kw: ZoonomiaOrthologSampler(seed=seed, **kw),
            adapter=lambda s, n, ctx: s.generate(n, task=ctx.task),
            assets=("zoonomia_alignment",),
            params={
                "distance": Param(
                    "all",
                    "evolutionary tier by identity to human: near | mid | far | all",
                    choices=("near", "mid", "far", "all"),
                ),
                "min_called_frac": Param(
                    0.95, "reject a window/species below this called (non-gap) fraction"
                ),
                "max_species_per_window": Param(
                    None, "cap species per window so a few loci cannot dominate"
                ),
                "dedupe": Param(
                    True,
                    "emit each distinct sequence once; a conserved window otherwise "
                    "returns the same 200-mer from many species (~15% of a raw draw)",
                ),
            },
        )
    )


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
    from albench.acquisition.uncertainty_variants import (
        ActivityNormalisedUncertaintyAcquisition,
        AggregateUncertaintyAcquisition,
        DifferentialUncertaintyAcquisition,
    )

    _acq(
        "uncertainty_aggregate",
        AggregateUncertaintyAcquisition,
        "Total variance summed across cell types. Variances add, uncertainties do "
        "not. Needs a multitask student; raises rather than collapsing to the "
        "single-condition rule.",
        "uncertainty",
    )
    _acq(
        "uncertainty_differential",
        DifferentialUncertaintyAcquisition,
        "|log variance ratio| between two cell types. Ratio not difference: variance "
        "is strictly positive, so a ratio is scale-free and symmetric in log space. "
        "Needs a multitask student.",
        "uncertainty",
    )
    _acq(
        "uncertainty_activity_normalised",
        ActivityNormalisedUncertaintyAcquisition,
        "Uncertainty EXCESS over the trend at that activity level. Raw uncertainty "
        "correlates with activity in regression, so ranking by it largely re-ranks by "
        "activity; this bins by predicted activity, takes a robust centre and spread "
        "per bin, and scores the standardised residual -- the Pareto frontier of "
        "uncertain-for-their-activity rather than merely active.",
        "uncertainty",
    )

    _acq(
        "badge_epistemic_only",
        EpistemicOnlyAcquisition,
        # Identical in EFFECT to uncertainty_mcdropout while the student's
        # epistemic_uncertainty() falls back to total uncertainty -- both are top-k
        # by the same score. They diverge only for a student that genuinely
        # separates epistemic from aleatoric (see separates_uncertainty()).
        # Report them as ONE arm until that is true.
        "CONTROL for BADGE: uncertainty without diversity. Currently the same rule "
        "as uncertainty_mcdropout -- see the note above.",
        "control",
    )
    _acq(
        "badge_kmeanspp_only",
        KMeansPPOnlyAcquisition,
        # Operates on the model's EMBEDDINGS, not raw sequence: it is BADGE's
        # k-means++ step with the uncertainty scaling removed, so it overlaps
        # heavily with diversity_kmeans. Put one of the two on a figure, not both,
        # or they read as independent evidence when they are not.
        "CONTROL for BADGE: diversity without uncertainty. Embedding-based and "
        "largely redundant with diversity_kmeans -- see the note above.",
        "control",
    )


def _bootstrap() -> None:
    """Register everything. Import failures name the strategy that could not load."""
    for fn in (
        _reg_zoonomia,
        _reg_encode_accessibility,
        _reg_mutagenesis,
        _reg_evoaug,
        _reg_motif,
        _reg_baselines,
        _reg_sheet_additions,
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
