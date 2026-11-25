"""Arm A3 continued: push the one reservoir parameter that actually moved the needle.

Arm A3's 48 variants produced exactly one large, clean effect, and it is not the one the
arm was designed around. Planting DENSITY is monotone in the designed endpoint, against
the motif_ct_enriched base (stacked r, 3 seeds, 30k, e500_wd2):

    mot_ct_n1_2     1-2 sites    designed 0.4633  (-0.0768)
    [base]          3-7 sites    designed 0.5401
    mot_ct_n8_15    8-15 sites   designed 0.5611  (+0.0211)
    mot_ct_n15_25   15-25 sites  designed 0.6171  (+0.0770)   <- and random +0.0129 too

Nothing else in the table comes close: every site-QUALITY knob loses (mutation rate 0.05
-0.044, 0.15 -0.060; vocabulary 30 -0.065, 120 -0.026; pure consensus -0.008), every tier
mixture loses, and the shared_core/ct_enriched blend (`mot_core_ct*`, ct_bias 0.2/0.5) is
flat to within noise against shared_core once it is scored against the right base.

Two things follow, and neither is answered by anything that has run:

1. THE LADDER HAS NOT PEAKED. 15-25 is the top rung that exists and it is still climbing.
   25-40 and 40-60 say where it turns over -- and it must turn over, since a 200-bp window
   cannot hold 60 non-overlapping 9-mers, so the top rung is really "overlapping sites",
   which is a different sequence distribution rather than more of the same one.
2. DENSITY WAS ONLY EVER TESTED ON THE WEAKER VOCABULARY. Every density row is
   motif_ct_enriched, but motif_shared_core is the better reservoir on this endpoint
   (0.6295 vs 0.5401). If the two effects compose, shared_core at 15-25 sites is the best
   corpus in the whole study and nobody has generated it. If they do not compose, that is
   the more interesting result, because it says density is substituting for vocabulary
   quality rather than adding to it.

Names avoid `_d`: bench33_legnet splits its pool stem on the FIRST `_d`, so a name
containing one silently collapses two variants into one output directory.
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

# name | strategy | params | why
VARIANTS: list[tuple[str, str, dict, str]] = [
    (
        "mot_sc_n15_25",
        "motif_shared_core",
        {"min_motifs": 15, "max_motifs": 25},
        "the composition test: the best vocabulary at the best density. If the two "
        "effects add, this is the best corpus in the study",
    ),
    (
        "mot_sc_n8_15",
        "motif_shared_core",
        {"min_motifs": 8, "max_motifs": 15},
        "the midpoint of the same ladder on shared_core, so the shape can be compared "
        "with ct_enriched's rather than only its endpoint",
    ),
    (
        "mot_ct_n25_40",
        "motif_ct_enriched",
        {"min_motifs": 25, "max_motifs": 40},
        "where the ct_enriched ladder turns over. 25-40 sites in 200bp forces overlap",
    ),
    (
        "mot_ct_n40_60",
        "motif_ct_enriched",
        {"min_motifs": 40, "max_motifs": 60},
        "past the point a 200-bp window can hold: the control that proves the ladder is "
        "a density effect and not an artefact of asking for more",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int)
    ap.add_argument("--field", default="all")
    a = ap.parse_args()

    bad = sorted(n for n, *_ in VARIANTS if "_d" in n)
    if bad:
        raise SystemExit("names collapse under stem.split('_d')[0]: " + ", ".join(bad))
    if len({n for n, *_ in VARIANTS}) != len(VARIANTS):
        raise SystemExit("duplicate variant names: two rows would share one output dir")

    name, strategy, params, why = VARIANTS[a.index]

    from albench.core.pools import PoolSpec

    spec = PoolSpec(
        strategy=strategy, params=dict(params), size=SIZE, generation_seed=GEN_SEED, task="k562"
    )
    fields = {
        "name": name,
        "strategy": strategy,
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
