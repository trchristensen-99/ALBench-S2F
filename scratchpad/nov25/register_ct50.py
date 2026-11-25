"""ct50: does the reservoir RANKING depend on which cell type you are training for?

This is the keystone question for the multi-cell half of the registered report and nothing
has ever answered it. Every reservoir comparison we have run is K562-only: K562 oracle,
K562 students, K562 endpoints. The 4M library will be measured in two or three cell lines,
so if the best reservoir for K562 is not the best for HepG2 the recommendation is a MIXTURE
and we need its ratio; if the ranking is stable, the 4M is one corpus and we can say so.

Three training targets per reservoir, all from the same 30k sequences:

    k562    the existing endpoint, re-run here so all three targets share one student path
    hepg2   the same sequences under the HepG2 oracle
    diff    trained directly on (K562 - HepG2). Training on the difference rather than
            subtracting two separately-trained models is the Nov-19 recommendation, and it
            is the model a cell-type-SPECIFIC design campaign actually needs.

The expensive part was already done and then dropped on the floor: `label_pool_multicell.py
relabel-reps` wrote HepG2 sidecars for all 11 reservoirs back in October, but
`build_hepg2_pools.py` had its strategy list hardcoded to the three the Oct-26 comparison
used, so eight reservoirs' oracle labels were computed and never turned into pools. They
are built now; this registers the students.

Everything below is copied from the `hepg2` block of scripts/program_oct22/register_oct26.py
without modification -- same menu, same shard count, same base_cap_min, same val caches,
same HP_ALLOW_UNSTAMPED -- so these rows and the existing nine hg30k_* cells are ONE table
rather than two experiments that happen to share an axis.

zoonomia_orthologs is deliberately absent: its r1 pool was rebuilt after the sidecar was
written (the validation-chromosome fix), so the builder's row-by-row guard refuses it, and
the reservoir is excluded from the comparison anyway.

    .venv/bin/python scratchpad/nov25/register_ct50.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
REG = f"{R}/scratchpad/dispatch_registry.json"
GP = f"{R}/outputs/grid_pools"
REPS = f"{GP}/k562_reps"
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"

# All ten, including the three the Oct-26 run already covered. Those three were trained
# with the then-current copy of menu_retrain.py, which has moved and changed since
# (cluster_live/ -> orchestration/), so mixing their rows with rows produced today would
# put a script version under one half of the table. Re-running them costs nine cell-triples
# and makes the whole thing one experiment; the old hg30k_* directories are left untouched.
NEW = [
    "genomic",
    "random",
    "motif_ct_enriched",
    "motif_shared_core",
    "motif_syntax_core",
    "evoaug_paper2023",
    "mutagenesis",
    "gc_matched",
    "dinuc_shuffle",
    "GRM",
]
D = 30_000
N_SHARDS = 3
BASE_CAP_MIN = 25  # CAP[30000] in register_oct26.py


def rel(p: str) -> str:
    return os.path.relpath(p, R)


def cells() -> list[dict]:
    out = []
    for tgt in ("k562", "hepg2", "diff"):
        for st in NEW:
            pool = (
                f"{REPS}/{st}_r1_d{D}.npz"
                if tgt == "k562"
                else f"{GP}/k562_hepg2/{tgt}/{st}_r1_d{D}.npz"
            )
            vc = f"outputs/val_variants/val_V0{'' if tgt == 'k562' else '_' + tgt}.npz"
            out_dir = f"outputs/menu_retrain_v0/ct50_{tgt}_{st}_r1"
            for k in range(N_SHARDS):
                out.append(
                    {
                        "name": f"ct50_{tgt}_{st}_s{k}",
                        "gpus": ["h100", "v100"],
                        "ext": [],
                        "out_dir": out_dir,
                        "env": {"HP_ALLOW_UNSTAMPED": "1"},
                        "needs": sorted({rel(pool), vc}),
                        "cmd": (
                            f"uv run --no-sync python orchestration/menu_retrain.py "
                            f"--menu {MENU} --reservoir_cache {pool} --D {D} "
                            f"--out_dir {R}/{out_dir} --shard {k}/{N_SHARDS} "
                            f"--base_cap_min {BASE_CAP_MIN} --val_cache {R}/{vc}"
                        ),
                        "done": {"type": "file", "path": f"{out_dir}/COMPLETE_{k}of{N_SHARDS}"},
                        "strategy": st,
                    }
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    new = cells()
    reg = json.load(open(REG))
    have = {c["name"] for c in reg}
    fresh = [c for c in new if c["name"] not in have]
    print(f"{len(new)} cells built, {len(new) - len(fresh)} already registered, {len(fresh)} new")
    print(f"estimated {len(fresh) * BASE_CAP_MIN / 60:.0f} GPU-h at the {BASE_CAP_MIN}-min cap")

    missing = sorted({n for c in fresh for n in c["needs"] if not os.path.exists(f"{R}/{n}")})
    for m in missing:
        print(f"  MISSING (held by the needs gate): {m}")

    if a.dry_run:
        print("\n--dry-run: registry not written")
        return

    os.replace(REG, REG + ".bak_ct50")
    with open(REG, "w") as fh:
        json.dump(reg + fresh, fh, indent=1)
    print(f"\nregistry: {len(reg)} -> {len(reg) + len(fresh)} cells (backup .bak_ct50)")


if __name__ == "__main__":
    main()
