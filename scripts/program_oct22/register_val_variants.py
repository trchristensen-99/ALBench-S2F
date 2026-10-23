"""Validation-set variant test: does the choice of validation set change what we conclude? (cells for the dispatcher; NOT launched.)

QUESTION. Early stopping, ElasticNet ensemble weights and the LLM proposer's feedback all use ONE validation set, real genomic
chr19/21/X sequences (V0). If that rewards genomic-like models, genomic-trained pools are flattered and the reservoir comparison is
biased. Variants (built by scripts/val_variants/build_val_variants.py, oracle_v3-labelled, ~29k sequences each, similarity-guarded
against every evaluation set and the training pools): V1 perturbed genomic, V2 diverse strategy families, V3 equal thirds of V0+V1+V2.

DESIGN. Retrain the 33-architecture menu (outputs/menu_retrain_v0/menu_v0.json, scratchpad/menu_retrain.py, shards k/3) on three pools at
D=30k, replicate r1 -- genomic, random, motif_ct_enriched (outputs/grid_pools/k562_reps) -- once per validation set. Arms per pool:

  V0   existing default-validation runs outputs/menu_retrain_v0/ind30k_{pool}_r1 (seeds r00-r02): baseline AND noise floor. Not re-run.
  V0x  default validation again (stored chr_val labels, no --val_cache), same init seeds as the other arms, plus --extra_val so every model also scores V0..V3.
  V0c  the SAME genomic chr19/21/X sequences with labels RE-SCORED by oracle_v3 (val_V0.npz). The stored chr_val_ref_only.npz labels are not oracle_v3
       (measured: r=0.983 with the oracle_v3 relabel, mean +0.067, GC-dependent offset 0.055-0.112) while every pool is, so V0x vs V0c isolates
       the label-space mismatch from the composition effect; V1-V3 are compared to V0c.
  V1/V2/V3  --val_cache val_Vk.npz (early stopping, val_pred, weights all from that set) plus --extra_val for all four sets.

--extra_val is read-only (it adds valpred_<name> arrays); it is what lets one trained model be scored against every validation set, which is
the only way to separate "which checkpoint did early stopping pick" from "how well does this validation score rank architectures" and
from "which set the ensemble weights were fitted on".

READOUT (scripts/val_variants/analyze_val_variants.py [--cross]):
  (a) test r on each of the 18 evaluation sets of the validation-ensembled result (ElasticNet fitted on that arm's validation predictions);
      paired against V0 round 0, in units of the V0 seed SD (3 seeds) so a difference smaller than training noise is not read as an effect;
  (b) Spearman across the 33 architectures between each validation score (own or cross-scored) and per-set test r, bootstrap CI over
      architectures; the ceiling is test-r ranking agreement between two seeds of the same arm;
  (c) epochs trained, share stopped by patience, and the paired single-model change in test r relative to V0.

COST (measured from ind30k_*_r1: train time only, plus ~10% for process start, data load and 18-set prediction):
  one arm x pool = 33 models ~ 2.2-2.3 H100-h  ->  ~2.5 H100-h  (~6.9 V100-h; the dispatcher falls back to V100 at 2.4-2.8x slower).
  Phase 1 (5 arms x 3 pools = 15 arm-pools = 45 shard cells of ~45 min H100 / ~2 h V100):        ~37 H100-h
  Phase 2 (--phase2: a second init seed, seed_offset 1, same out_dirs, shards k/4; 60 more cells): +37 H100-h
Cells carry the `ind30` name prefix so the dispatcher gives them the same 3.5 h slow_nice wall time as the other menu retrains (no dispatcher change).
"""

import json
import os
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
from stage2_lib import REPS, R  # noqa: E402

os.chdir(R)
PHASE2 = "--phase2" in sys.argv
DRY = "--dry" in sys.argv
MENU = f"{R}/outputs/menu_retrain_v0/menu_v0.json"
VV = f"{R}/outputs/val_variants"


def rel(p):
    return os.path.relpath(p, R)


HV = ["h100", "v100"]
POOLS = ["genomic", "random", "motif_ct_enriched"]
VALS = {v: f"{VV}/val_{v}.npz" for v in ("V0", "V1", "V2", "V3")}
EXTRA = ",".join(f"{v}={p}" for v, p in VALS.items())
for p in VALS.values():
    assert os.path.exists(p), (
        f"missing {p}: build the variants first (scripts/val_variants/build_val_variants.py)"
    )
for st in POOLS:
    assert os.path.exists(f"{REPS}/{st}_r1_d30000.npz"), st

reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}


def cell(arm, pool, k, seed_offset):
    """Phase 1 (seed_offset 0): shards k/3. Phase 2 (seed_offset 1): SAME out_dir, shards k/4 -- the done-marker COMPLETE_{k}of4 differs from phase 1's COMPLETE_{k}of3,
    the same trick the ind30x cells use, so the dispatcher does not mistake the second seed for finished work."""
    pooln = f"{REPS}/{pool}_r1_d30000.npz"
    out = f"outputs/menu_retrain_v0/vv30k_{arm}_{pool}_r1"
    n_sh = 3 if seed_offset == 0 else 4
    cache = "" if arm == "V0x" else f" --val_cache {VALS['V0' if arm == 'V0c' else arm]}"
    needs = [rel(pooln)] + [rel(VALS[v]) for v in ("V0", "V1", "V2", "V3")]
    return dict(
        name=f"ind30v_{arm}_{pool}_r1" + (f"_o{seed_offset}" if seed_offset else "") + f"_s{k}",
        gpus=HV,
        ext=[],
        out_dir=out,
        env={},
        needs=sorted(set(needs)),
        cmd=f"uv run --no-sync python scratchpad/menu_retrain.py --menu {MENU} --reservoir_cache {pooln} --D 30000 --out_dir {R}/{out} --shard {k}/{n_sh} --base_cap_min 25"
        f"{cache} --extra_val {EXTRA}" + (f" --seed_offset {seed_offset}" if seed_offset else ""),
        done=dict(type="file", path=f"{out}/COMPLETE_{k}of{n_sh}"),
    )


cells = []
for off in (0, 1) if PHASE2 else (0,):
    for pool in POOLS:  # pool-major so a whole pool's four arms finish together
        for arm in ("V0x", "V0c", "V1", "V2", "V3"):
            for k in range(3 if off == 0 else 4):
                cells.append(cell(arm, pool, k, off))
new = [c for c in cells if c["name"] not in have]
print(f"{len(cells)} cells ({len(new)} new); phase2={PHASE2}")
for c in new[:2]:
    print(json.dumps(c, indent=1))
if DRY:
    sys.exit(0)
reg = reg + new
json.dump(reg, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
print("registered", len(new), "menu-retrain shards; registry", len(reg))
