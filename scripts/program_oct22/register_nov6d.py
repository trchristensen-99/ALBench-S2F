"""Nov-6: the 300k replicate pools that are missing, and a test of whether the search loop's winners survive reseeding.

1. `pool300_<reservoir>_g<seed>` (9 cells): the 300k rung runs four reservoirs only because raw replicate pools exist for four (`k562_reps_raw300k` holds evoaug_paper2023, motif_ct_enriched and random at generation
   seeds 43/44/45, plus genomic as permutations of its finite set). `mutagenesis`, `motif_shared_core` and `zoonomia_orthologs` are reseed-mode strategies with a ceiling near 1M, so nothing stops them except that
   nobody generated them: this builds and oracle-labels three independent 300k draws of each, which is what "multiple draws as independent replicates" needs. ~2.5 GPU-hours of labelling plus generation.

2. `reg48_*` (non-LLM): does the loop actually find better architectures, or does it select lucky training runs? Its winners are chosen on ONE run each, so "best of 30" is biased upward by the seed noise it is
   selected on, while the bank's single-model figure is a mean over 20 seeds. Comparing them as they stand is unfair in the loop's favour, and the loop still loses (best single 0.8726 against the bank's 0.8748).
   This retrains the top 8 loop architectures by validation r, 5 seeds each under the tuned 300-epoch recipe, so each one gets a MEAN rather than a best-of-one. If their means fall well below their selected
   scores, the loop is ranking noise and its architectures are no better than LegNet; if they hold, the architectures are real and the gap is elsewhere.

    python scripts/program_oct22/register_nov6d.py [--dry]
"""

import fcntl
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_fx35b as fb  # noqa: E402
import register_nov2b as nb  # noqa: E402

R, NEST, DRY = nb.R, nb.NEST, nb.DRY
AR = "outputs/archsearch_reps_oct22"
MISSING = ("mutagenesis", "motif_shared_core", "zoonomia_orthologs")
GEN_SEEDS = {1: 43, 2: 44, 3: 45}  # the seeds the existing replicates used
TUNED = {"optimizer": "adamw", "lr": 0.01, "weight_decay": 0.2, "batch_size": 1024, "epochs": 300, "lr_schedule": "onecycle"}  # fmt: skip
ARMS = (
    "ar41",
    "ln41",
    "rr40",
    "is40",
)  # the fixed-harness arms whose models all trained 300 epochs


def pool_cells():
    out = []
    for res in MISSING:
        for rep, gseed in GEN_SEEDS.items():
            o = f"outputs/grid_pools/k562_reps_raw300k_new/{res}__g{gseed}"
            out.append(
                dict(
                    name=f"pool300_{res}_g{gseed}",
                    gpus=["h100", "v100"],
                    ext=[],
                    out_dir=o,
                    env={},
                    needs=[],
                    cmd=f"uv run --no-sync python scripts/build_pool.py --strategy {res} --size 300000 --generation-seed {gseed} --out-dir {R}/{o} && touch {R}/{o}/COMPLETE",
                    done=dict(type="file", path=f"{o}/COMPLETE"),
                )
            )
    return out


def winners(n=8):
    """The top distinct loop architectures by validation r, one per idea, from the arms that trained a full schedule."""
    cand = []
    for arm in ARMS:
        for d in sorted(glob.glob(f"{R}/{AR}/genomic_30k_{arm}/rep*")):
            for f in glob.glob(f"{d}/r*_arch_*_meta.json"):
                m = json.load(open(f))
                if m.get("epochs_trained", 0) >= 290 and m.get("val_pearson"):
                    cand.append((float(m["val_pearson"]), arm, os.path.basename(d), m))
    cand.sort(key=lambda x: -x[0])
    picked, seen = [], set()
    for val, arm, rep, m in cand:
        key = m["idea"][:60]
        if key in seen:
            continue
        seen.add(key)
        picked.append(
            {
                "id": f"w_{arm}_{rep}_{m['model_id']}",
                "cell": f"{arm}/{rep}",
                "idea": m["idea"],
                "orig_val": val,
                "n_params": int(m["n_params"]),
                "source": m["source"],
                "train_config": dict(TUNED),
            }
        )
        if len(picked) == n:
            break
    return picked


def reg_cells():
    ents = winners()
    assert len(ents) == 8, len(ents)
    path = f"{R}/outputs/menus_nov6/WINNERS_30k.json"
    if not DRY:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(ents, open(path, "w"), indent=1)
    for e in ents:
        print(f"    {e['orig_val']:.4f}  {e['cell']:12s}  {e['idea'][:62]}")
    return fb.cells(
        "reg48_winners_30k", path, f"{NEST}/genomic_d30000.npz", 30000,
        {"AL_PATIENCE": "9999", "AL_PRECISION": "fp16", "AL_RC_MODE": "aug"}, nb.STD, shards=5, cap=90, seeds=5,
    )  # fmt: skip


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in pool_cells() + reg_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print(new[0]["cmd"][:240])
            return
        shutil.copy(path, path + ".bak_nov6d")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
