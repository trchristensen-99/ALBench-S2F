"""The reservoir-PARAMETER sweep: one table, read by the pool builder and the registrar.

Arm A answered "which reservoir family", at 19 families x 30k x 5 seeds. It did not
answer "which settings", and for a library that is the live half of the question: the
collaborators are not going to order "motif-planted sequences", they are going to order
sequences with some number of sites of some strength from some vocabulary, and nothing
measured so far distinguishes those.

Five axes, each with its own baseline already in hand from Arm A, so every row here is
a difference against a measured number rather than a standalone score:

  DENSITY      how many motifs per 200-mer                 (Arm A ran 3-7)
  VOCABULARY   how many distinct motifs to draw from       (Arm A ran the full ~1,021)
  STRENGTH     consensus vs PWM-sampled vs mutated sites   (Arm A ran pwm_sample, 0.0)
  RATE         per-base mutation rate, fixed or spread     (Arm A ran 0.05 and 0.10)
  OPERATION    which EvoAug edit does the work             (Arm A ran the default mix)

plus the model-guided objectives, which are a different kind of axis: not "how do we
perturb" but "what are we aiming at". Those are the arms registered on Nov 24 in
albench/core/registry.py, and every one of them is read against `mutagenesis` at the
same per-base rate rather than against `random`, because the evolution loop costs oracle
calls that random mutagenesis does not.

One deliberate omission: no arm here changes two axes at once. A 2-axis grid over five
axes is 10 cells per pair and we have one night.
"""

from __future__ import annotations

# (readable name, strategy, {param: value}, why this row exists)
VARIANTS: list[tuple[str, str, dict[str, object], str]] = [
    # ---- DENSITY: sites per sequence -------------------------------------------
    (
        "mot_ct_n1_2",
        "motif_ct_enriched",
        {"min_motifs": 1, "max_motifs": 2},
        "sparse planting: one or two sites in 200bp, close to a real promoter",
    ),
    (
        "mot_ct_n8_15",
        "motif_ct_enriched",
        {"min_motifs": 8, "max_motifs": 15},
        "dense planting: more sites than a real CRE carries",
    ),
    (
        "mot_ct_n15_25",
        "motif_ct_enriched",
        {"min_motifs": 15, "max_motifs": 25},
        "saturated: the point where sites must overlap, so syntax is forced",
    ),
    # ---- VOCABULARY: how many distinct motifs ----------------------------------
    (
        "mot_ct_v30",
        "motif_ct_enriched",
        {"vocab_size": 30},
        "narrow vocabulary, the size at which pairwise coverage is adequate",
    ),
    (
        "mot_ct_v120",
        "motif_ct_enriched",
        {"vocab_size": 120},
        "the upper end of adequate pair coverage (docs put the limit near 120)",
    ),
    # ---- STRENGTH: how good a binding site ------------------------------------
    (
        "mot_ct_mut05",
        "motif_ct_enriched",
        {"motif_mutation_rate": 0.05},
        "weak sites alongside strong ones -- Peter's 2026-09-12 request, never run",
    ),
    (
        "mot_ct_mut15",
        "motif_ct_enriched",
        {"motif_mutation_rate": 0.15},
        "mostly-degraded sites: where does a planted site stop being a site",
    ),
    (
        "mot_ct_consensus",
        "motif_ct_enriched",
        {"plant_mode": "consensus"},
        "every site the identical consensus string: the MEMORISATION control, and the "
        "cheapest way to find out whether the motif arms are teaching a lookup table",
    ),
    # ---- VOCABULARY SET: which motifs, not how many ----------------------------
    (
        "mot_core_ct20",
        "motif_shared_core",
        {"ct_bias": 0.2},
        "shared core upweighted 20% toward cell-type-enriched TFs: a blend rather than "
        "a choice between the two motif arms that lead Arm A on different endpoints",
    ),
    (
        "mot_core_ct50",
        "motif_shared_core",
        {"ct_bias": 0.5},
        "the same blend at half-and-half, so the blend axis has two points",
    ),
    # ---- RATE: how hard to mutate ----------------------------------------------
    (
        "mut_r002",
        "mutagenesis",
        {"mutation_rate": 0.02},
        "below the 0.05 Arm A ran: close enough to the parent to stay a real CRE",
    ),
    (
        "mut_r20",
        "mutagenesis",
        {"mutation_rate": 0.20},
        "far above it: 40 substitutions per 200-mer, most motifs destroyed",
    ),
    (
        "mut_ladder",
        "mutagenesis",
        {
            "mutation_rate_distribution": "uniform",
            "min_rate": 0.01,
            "max_rate": 0.20,
        },
        "ONE pool spanning the whole rate range instead of committing to a rate. This "
        "is the arm that matters for the library: if the ladder beats every fixed rate, "
        "the recommendation is a spread, and if it lands between them it is a dilution",
    ),
    # ---- OPERATION: which EvoAug edit does the work ----------------------------
    # Single-op arms, so a gain in the mixed arm can be attributed. Arm A measured the
    # mix and the published settings and could not say which operation earned it.
    (
        "ea_indel_only",
        "evoaug",
        {
            "p_deletion": 0.5,
            "p_insertion": 0.5,
            "p_inversion": 0.0,
            "p_translocation": 0.0,
            "p_tandem_dup": 0.0,
            "p_point_mutation": 0.0,
        },
        "length-changing edits alone",
    ),
    (
        "ea_inv_only",
        "evoaug",
        {
            "p_deletion": 0.0,
            "p_insertion": 0.0,
            "p_inversion": 1.0,
            "p_translocation": 0.0,
            "p_tandem_dup": 0.0,
            "p_point_mutation": 0.0,
        },
        "inversion alone: preserves composition exactly, destroys strand and order",
    ),
    (
        "ea_trans_only",
        "evoaug",
        {
            "p_deletion": 0.0,
            "p_insertion": 0.0,
            "p_inversion": 0.0,
            "p_translocation": 1.0,
            "p_tandem_dup": 0.0,
            "p_point_mutation": 0.0,
        },
        "translocation alone: preserves content, changes spacing -- the syntax edit",
    ),
    (
        "ea_dup_only",
        "evoaug",
        {
            "p_deletion": 0.0,
            "p_insertion": 0.0,
            "p_inversion": 0.0,
            "p_translocation": 0.0,
            "p_tandem_dup": 1.0,
            "p_point_mutation": 0.0,
        },
        "tandem duplication alone: our op, not the paper's, so it needs its own row",
    ),
    (
        "ea_point_only",
        "evoaug",
        {
            "p_deletion": 0.0,
            "p_insertion": 0.0,
            "p_inversion": 0.0,
            "p_translocation": 0.0,
            "p_tandem_dup": 0.0,
            "p_point_mutation": 1.0,
            "point_mutation_rate": 0.02,
        },
        "substitutions alone, which should reproduce `mutagenesis` at 0.02 -- and if it "
        "does not, the two samplers disagree about what a point mutation is",
    ),
    (
        "ea_equal",
        "evoaug",
        {
            "p_deletion": 0.2,
            "p_insertion": 0.2,
            "p_inversion": 0.2,
            "p_translocation": 0.2,
            "p_tandem_dup": 0.2,
            "p_point_mutation": 0.2,
        },
        "all operations equally likely: the mix with no tuning in it",
    ),
    (
        "ea_indel5",
        "evoaug",
        {"max_indel_size": 5},
        "small edits only: magnitude held down with the op mix at its default",
    ),
    (
        "ea_indel50",
        "evoaug",
        {"max_indel_size": 50},
        "quarter-length edits: magnitude as its own axis, separate from which op",
    ),
    # ---- OBJECTIVE: model-guided, aimed at different parts of the range ---------
    # Registered Nov 24. These cost oracle calls per generation, so each is read
    # against `mutagenesis` at the same rate, never against `random`.
    (
        "ise_minimize",
        "ise_minimize",
        {},
        "evolve toward silencing: the half of the range real CREs never occupy",
    ),
    (
        "ise_target_mid",
        "ise_target_mid",
        {},
        "evolve toward mid activity, where the assay's own ceiling is ~0",
    ),
    (
        "ise_diverse",
        "ise_diverse",
        {},
        "lineages aimed uniformly across the range: the 'cover the dynamic range' arm",
    ),
    (
        "ise_explore",
        "ise_explore",
        {},
        "maximise at temperature 4: separates the objective from greedy selection",
    ),
    # ---- OBJECTIVE, but guided by a BETTER model --------------------------------
    # The same two objectives again with a guide trained on 300k instead of 30k. This is
    # the axis that decides whether model-guided design is worth proposing at all: if a
    # 10x better guide does not make guided design beat mutagenesis, the method is not
    # limited by the guide and more data will not rescue it.
    (
        "ise_maximize_g300k",
        "ise_maximize",
        {},
        "maximise, guided by the 300k bank rather than the 30k one",
    ),
    (
        "ise_diverse_g300k",
        "ise_diverse",
        {},
        "spread across the range, guided by the 300k bank: the guide-quality contrast",
    ),
    (
        "ise_maximize_g30k",
        "ise_maximize",
        {},
        "the 30k-guided maximisation baseline the two g300k rows are read against",
    ),
]

# Which trained bank guides each model-in-the-loop arm. Guiding with the ORACLE would
# make these upper bounds -- selection at the argmax of the very ensemble that then
# labels the pool, which no collaborator can reproduce -- and would also cost ~14 GPU-h
# per pool against ~25 minutes for a LegNet bank. Arm A's saved weights are the point:
# these are the models we already have, which is the situation a collaborator is in.
GUIDES: dict[str, str] = {
    "ise_minimize": "outputs/wed41/bench38_ln_e500_wd2_30k_c20",
    "ise_target_mid": "outputs/wed41/bench38_ln_e500_wd2_30k_c20",
    "ise_diverse": "outputs/wed41/bench38_ln_e500_wd2_30k_c20",
    "ise_explore": "outputs/wed41/bench38_ln_e500_wd2_30k_c20",
    "ise_maximize_g30k": "outputs/wed41/bench38_ln_e500_wd2_30k_c20",
    "ise_maximize_g300k": "outputs/wed41/bench38_ln_e300_wd05_300k_c20",
    "ise_diverse_g300k": "outputs/wed41/bench38_ln_e300_wd05_300k_c20",
}

# Arm A rows these are differences against. Not generated here -- they already exist in
# outputs/grid_pools/k562_nested and were trained with 5 seeds in outputs/wed41.
BASELINES = {
    "motif_ct_enriched": "motif_ct_enriched (3-7 sites, full vocab, pwm_sample, mut 0.0)",
    "motif_shared_core": "motif_shared_core (ct_bias 0.0)",
    "mutagenesis": "mutagenesis (rate 0.05) and mutagenesis_mr10 (rate 0.10)",
    "evoaug": "evoaug (tuned mix), evoaug_ours_default, evoaug_paper2023",
    "ise": "mutagenesis at the matched per-base rate, NOT random",
}


def as_set_args(params: dict[str, object]) -> list[str]:
    """Render a parameter dict as the --set KEY=VALUE arguments build_pool.py takes."""
    return [f"--set {k}={v}" for k, v in sorted(params.items())]


if __name__ == "__main__":
    print(f"{len(VARIANTS)} variant pools over {len({v[1] for v in VARIANTS})} strategies\n")
    for name, strat, params, why in VARIANTS:
        print(f"{name:20s} {strat:20s} {params}")
        print(f"{'':20s} {why}")
