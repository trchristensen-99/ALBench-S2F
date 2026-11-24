"""Arm A3: motif TIERS and gradient design. The table, and its build arguments.

Two families, both added on 2026-11-24 in response to the same objection -- that a
reservoir which only ever plants motifs believed active cannot teach a model what an
inactive motif looks like, and cannot discover that an "inactive" annotation is wrong.

TIER ARMS. Read against motif_ct_enriched and motif_shared_core from Arm A, and against
each other. `tier_active_only` is the control that matters: it is what we would have built
without the objection -- shared plus cell-type-specific, nothing negative -- so the gap
between it and `tier_balanced` is the value of including sites the annotation calls dead.
If that gap is zero, the annotation tiers are not worth library space and we should say so
rather than quietly shipping them.

All tier arms run with native_min_len=8 (the registry default for this strategy). Without
it the unrestricted 628-entry vocabulary blocks its own planting and realises 2.21 sites
per sequence against the 3-7 asked, while the arms it is compared against realise ~5 --
the comparison would be measuring density, not tiers.

FASTSEQPROP ARMS. Guided by Arm A's own 30k genomic bank, not the oracle, for the reason
in albench/ensemble/bank.py -- and more sharply here than for the evolutionary arms,
because gradient ascent is much better at finding a model's adversarial maxima than
mutation is. The canary measured exactly that: the guide scored its own designs 14.08
while the oracle scored them 6.92, against 0.48 for the CREs they started from. So the
design is real -- the oracle is independent of the guide and still sees a large gain --
but about half the claimed improvement is the designer exploiting its guide, and the
guide-oracle gap is the number to report for every one of these arms.

`fsp_target` exists because 6.92 is far outside the measured range (motif pools sit at
0.78-1.21 and epistemic selection reaches 1.94-2.38). A corpus of sequences nobody can
measure teaches a model about a regime the assay cannot report, so the mid-range designer
is the one most likely to be useful and the maximiser is partly a diagnostic.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

RAW_DIR = "outputs/grid_pools/k562_variants_raw"
SIZE = 30_000
GEN_SEED = 42

# The 30k genomic bank from Arm A: a student trained on data we already have, which is
# what a collaborator would actually be able to use as a designer.
BANK = "outputs/wed41/bench38_ln_e500_wd2_30k_c20"

# name | strategy | params | why
VARIANTS: list[tuple[str, str, dict, str]] = [
    (
        "tier_balanced",
        "motif_tiered",
        {"tier_weights": "shared_core=0.35,k562_only=0.30,hepg2_only=0.20,decoy=0.15"},
        "all four classes present: the arm the objection asks for",
    ),
    (
        "tier_active_only",
        "motif_tiered",
        {"tier_weights": "shared_core=0.50,k562_only=0.50"},
        "CONTROL: what we would have built anyway. The gap to tier_balanced is the "
        "value of planting sites the annotation calls inactive",
    ),
    (
        "tier_nodecoy",
        "motif_tiered",
        {"tier_weights": "shared_core=0.40,k562_only=0.35,hepg2_only=0.25"},
        "isolates the decoy contribution: real-but-inactive motifs without any "
        "non-functional sequence",
    ),
    (
        "tier_heavy_inactive",
        "motif_tiered",
        {"tier_weights": "shared_core=0.25,k562_only=0.25,hepg2_only=0.35,decoy=0.15"},
        "negative evidence dominant. If discrimination is what matters, more of it "
        "should help; if the corpus just needs active sites, this should lose",
    ),
    (
        "tier_with_other",
        "motif_tiered",
        {"tier_weights": ("shared_core=0.30,k562_only=0.25,hepg2_only=0.20,other=0.15,decoy=0.10")},
        "adds the UNANNOTATED tier: motifs in the vocabulary that no cell-type set "
        "lists. The direct probe for annotation incompleteness",
    ),
    (
        "tier_fine_vocab",
        "motif_tiered",
        {
            "tier_weights": "shared_core=0.35,k562_only=0.30,hepg2_only=0.20,decoy=0.15",
            "vocab_max_len": 20,
            "vocab_trim_ic": 0.0,
        },
        "same mixture over a WIDER vocabulary (long, low-IC motifs kept). Tests "
        "whether the 12-bp/0.5-IC filter is discarding usable binding preferences",
    ),
    (
        "fsp_max",
        "fastseqprop",
        {},
        "gradient design toward high activity, from real CREs",
    ),
    (
        "fsp_target",
        "fastseqprop_target",
        {},
        "gradient design toward the MIDDLE of the range -- the one most likely to "
        "stay inside what the assay can measure",
    ),
    (
        "fsp_min",
        "fastseqprop_min",
        {},
        "gradient design toward LOW activity: the bottom of the range, which no "
        "reservoir currently targets",
    ),
    (
        "fsp_rand_init",
        "fastseqprop_random_init",
        {},
        "CONTROL for fsp_max: designs from noise rather than from a real CRE, so the "
        "starting sequence's contribution is separable from the optimisation's",
    ),
]

GUIDES: dict[str, str] = {n: BANK for n, s, _, _ in VARIANTS if s.startswith("fastseqprop")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int, nargs="?", default=0)
    ap.add_argument(
        "--field",
        default="all",
        choices=("all", "name", "strategy", "setargs", "path", "why", "guide"),
    )
    ap.add_argument("--count", action="store_true")
    a = ap.parse_args()
    if a.count:
        print(len(VARIANTS))
        return
    if not 0 <= a.index < len(VARIANTS):
        raise SystemExit(f"index {a.index} out of range: {len(VARIANTS)} variants")

    # bench33_legnet.py derives its output directory as stem.split("_d")[0], splitting on
    # the FIRST "_d", so a name containing "_d" collapses and two variants can silently
    # share one directory and overwrite each other's weights. This cost nothing to check
    # and caught three names on Nov 24.
    bad = sorted(n for n, *_ in VARIANTS if "_d" in n)
    if bad:
        raise SystemExit("names collapse under stem.split('_d')[0]: " + ", ".join(bad))
    if len({n for n, *_ in VARIANTS}) != len(VARIANTS):
        raise SystemExit("duplicate variant names: two rows would share one output dir")

    name, strategy, params, why = VARIANTS[a.index]

    from albench.core.pools import PoolSpec
    from albench.core.registry import get

    guide = GUIDES.get(name)
    ident = dict(params)
    if "oracle" in (get(strategy).assets or ()):
        ident["guide_model"] = f"bank:{os.path.basename(guide)}" if guide else "oracle:ag_s2"

    spec = PoolSpec(
        strategy=strategy, params=ident, size=SIZE, generation_seed=GEN_SEED, task="k562"
    )
    fields = {
        "name": name,
        "strategy": strategy,
        "guide": guide or "",
        "setargs": "\n".join(f"--set\n{k}={v}" for k, v in sorted(params.items())),
        "path": str(spec.path(RAW_DIR)),
        "why": why,
    }
    if a.field == "all":
        for k in ("name", "strategy", "path", "why"):
            print(f"{k}\t{fields[k]}")
        return
    out = fields[a.field]
    if out:
        print(out)


if __name__ == "__main__":
    main()
