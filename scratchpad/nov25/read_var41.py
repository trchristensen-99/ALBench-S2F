"""The Arm A3 parameter table: 48 reservoir VARIANTS at 30k, read against their families.

Arm A ranks reservoir FAMILIES (motif, evoaug, mutagenesis, model-based, random, genomic).
Arm A3 asks the next question down: within a family, which setting of its parameters? Every
row here is the same student -- LegNet at the tuned 30k recipe (e500_wd2), three seeds,
240-bp standard context -- on 30,000 oracle-labelled sequences, so the only thing that
varies is how those 30,000 were generated.

Grouped by family rather than printed flat, because the decision the registered report has
to make is "which member of each family do we test", and a flat ranking across families
hides that the within-family spread is sometimes larger than the between-family one.

Baselines are the canonical Arm A 30k cells, printed at the top of each family so a
variant's delta is against its own family's default rather than against the best row.

    python scratchpad/nov25/read_var41.py [--sort designed]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "orchestration"))
sys.path.insert(0, str(REPO / "scripts/analysis"))
from read_acq52 import SETS, read_run  # noqa: E402

VAR = REPO / "outputs/var41"
MR = REPO / "outputs/menu_retrain_v0"
PREFIX = "bench38_ln_e500_wd2_30k_c20"
PRETTY = {"genomic": "genomic", "ood": "designed", "random_32k": "random"}

# family -> (baseline cell under menu_retrain_v0, [variant suffixes under outputs/var41])
FAMILIES: dict[str, tuple[str | None, list[str]]] = {
    "motif (tiers: which annotation classes to plant)": (
        "motif_ct_enriched",
        [
            "tier_balanced",
            "tier_active_only",
            "tier_nodecoy",
            "tier_heavy_inactive",
            "tier_with_other",
            "tier_fine_vocab",
            "tier_up_ct",
            "tier_up_shared",
            "tier_flat",
            "tseq_pure",
            "tseq_half",
            "tseq_mixed",
        ],
    ),
    "motif (vocabulary and planting density)": (
        "motif_ct_enriched",
        [
            "mot_core_ct20",
            "mot_core_ct50",
            "mot_ct_v30",
            "mot_ct_v120",
            "mot_ct_n1_2",
            "mot_ct_n8_15",
            "mot_ct_n15_25",
            "mot_ct_consensus",
            "mot_ct_mut05",
            "mot_ct_mut15",
        ],
    ),
    "evoaug (which perturbation)": (
        "evoaug",
        [
            "ea_equal",
            "ea_point_only",
            "ea_inv_only",
            "ea_trans_only",
            "ea_tandem_only",
            "ea_indel_only",
            "ea_indel5",
            "ea_indel50",
        ],
    ),
    "mutagenesis (rate)": ("mutagenesis", ["mut_r002", "mut_r20", "mut_ladder"]),
    "model-based design (gradient and search)": (
        None,
        [
            "fsp_max",
            "fsp_target",
            "fsp_min",
            "fsp_rand_init",
            "sa_max",
            "sa_target",
            "ada_max",
            "ada_target",
        ],
    ),
    "model-based selection (ISE objective)": (
        None,
        [
            "ise_maximize_g30k",
            "ise_maximize_g300k",
            "ise_minimize",
            "ise_target_mid",
            "ise_spread",
            "ise_spread_g300k",
            "ise_explore",
        ],
    ),
}


def row(d: Path):
    if not d.is_dir():
        return None
    return read_run(d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default=None, choices=[*SETS, "designed", "none"])
    a = ap.parse_args()
    key = {"designed": "ood"}.get(a.sort, a.sort)

    for fam, (base, variants) in FAMILIES.items():
        print(f"\n=== {fam} ===")
        hdr = f"{'arm':<26}{'K':>3}" + "".join(f"{PRETTY[s]:>12}" for s in SETS)
        print(hdr)
        print("-" * len(hdr))
        ref: dict[str, float] = {}
        if base:
            got = row(MR / f"{PREFIX}_{base}")
            if got:
                _, stack, k = got
                ref = stack
                cells = "".join(f"{stack[s]:>12.4f}" for s in SETS)
                print(f"{'[base] ' + base:<26}{k:>3}{cells}")
        out = []
        for v in variants:
            got = row(VAR / f"{PREFIX}_{v}")
            if got is None:
                out.append((None, v, None, 0))
                continue
            _, stack, k = got
            out.append((stack.get(key) if key and key != "none" else None, v, stack, k))
        if key and key != "none":
            out.sort(key=lambda t: (t[0] is None, -(t[0] or 0)))
        for _, v, stack, k in out:
            if stack is None:
                print(f"{v:<26}  --  (no finished members)")
                continue
            cells = ""
            for s in SETS:
                d = stack[s] - ref[s] if ref else None
                cells += f"{stack[s]:>12.4f}" if d is None else f"{stack[s]:>8.4f}{d:>+8.4f}"
            print(f"{v:<26}{k:>3}{cells}")
    print("\nstacked (enet_all_pos) r; +-value is the delta against that family's base cell.")
    print("seed noise on this harness is ~0.008 -- a delta inside that is not a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
