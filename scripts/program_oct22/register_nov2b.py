"""Nov-2 evening: follow-ups to the lab-meeting feedback, as dispatcher cells (the gated groups are NOT launched until released).

HARNESS FIX FIRST. The launcher every loop cell runs (scratchpad/arch_autoresearch.py) stopped each model after 8 epochs without improvement, hard-coded: 90% of the 1,498 models of the standard-input loop runs stop before 90% of the
schedule they asked for (100% at 10k) and a model that asks for >= 200 epochs trains a median of 26 to 36, so the tuned LegNet recipe (one-cycle, e500 to e800) could not run and every comparison with the seed bank was between
a fully trained model and one cut at the high-learning-rate phase. patch_loop_patience.py makes the launcher read AL_PATIENCE (default 8: nothing earlier changes) and AL_ONECYCLE_FULL=1 (a one-cycle schedule is never stopped by patience).
The "39" arms below run under H2 = AL_PATIENCE 16 (the menu standard), AL_ONECYCLE_FULL 1 and AL_PRECISION fp16 (what the bank trains with; the harness default is fp32), and longer per-model caps.

  pa38_ref_<D>k   harness parity (menu library, patience off, fp16): the published MPRA-LegNet (albench.arch_search.mpra_legnet_ref.SOURCE) with the bank's recipes (AdamW lr 0.01, batch 1024, one-cycle; e100 and the per-size tuned
                  epochs / weight decay), 3 seeds, against the native bank bench38_ln_* (same architecture and recipe, its own trainer).
  rr39_30k_r<k>   the standard managed loop (rr38 settings: coordinator, 40% explore, reference LegNet and bank bar, 15 rounds x 2) under H2, 3 replicate pools: does the loop reach the single-model level of the tuned LegNet now?
  lv39_<D>k_r<k>  the loop restricted to LegNet variants (manager "variants": small modifications of the tuned MPRA-LegNet and of the best variant so far, no new mechanisms), under H2; 8 rounds x 2 (6 at 100k), 10k x2, 30k x3, 100k x1.
  ef39_<ef>_30k_r<k>  explore fraction 0.2, 0.6 and 1.0 with the rr39 settings (rr39 is the 0.4 arm). GATED (ef39_go).
  vc39_30k_r<k>   the rr39 settings with the mixed validation set val_B1 (scripts/val_variants/build_val_composition.py). GATED (vc39_go).

Gates are files the cell `needs` (scratchpad/gates/<name>_go); `touch` one to release a group (lv39b_go: the 10k and 100k variant cells, staged after the account limit was hit with 11 loop cells).

    python scripts/program_oct22/register_nov2b.py [--dry] [--only pa38,lv39,rr39,ef39,vc39]
"""

import copy
import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_fx35b as fb  # noqa: E402
import register_lr35 as lr  # noqa: E402
import register_oct30 as r30  # noqa: E402

R, NEST = r30.R, r30.NEST
DRY = "--dry" in sys.argv
ONLY = (
    sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else "pa38,lv39,rr39,ef39,vc39"
).split(",")
GATES = "scratchpad/gates"
STD = " --flank_ctx 20 --shift_max 10"
# per-size tuned recipe of the seed banks (docs/TUNED_LEGNET.md): epochs, weight decay
TUNED = {10000: (800, 0.4), 30000: (500, 0.2), 100000: (500, 0.1)}
COORD_VARIANTS = {
    "stages": ["propose", "coordinate", "code", "evaluate"],
    "ideas_per_round": 6,
    "manager": "variants",
}
RR = {
    "stages": ["propose", "coordinate", "code", "evaluate"],
    "ideas_per_round": 6,
    "manager": "schedule_ens",
}
LV_SIZES = {
    10000: (2, 8, 30, 12.0),
    30000: (3, 8, 60, 24.0),
    100000: (1, 6, 100, 36.0),
}  # reps, rounds, per-model cap (H100-min), budget hours
RR_SIZE = (3, 40, 16.0)  # reps, per-model cap, budget hours at 30k (rr38: 25 and 10.5)
H2 = {
    "AL_PATIENCE": "16",
    "AL_ONECYCLE_FULL": "1",
    "AL_PRECISION": "fp16",
}  # the fixed harness (see the docstring)


def parity_menu(D: int) -> str:
    """Menu of the published MPRA-LegNet under the bank's two recipes at this size; written once, read by the retrain cells."""
    from albench.arch_search.mpra_legnet_ref import SOURCE

    ep, wd = TUNED[D]
    base = {"optimizer": "adamw", "lr": 0.01, "batch_size": 1024, "lr_schedule": "onecycle"}
    entries = [
        {"id": f"legnet_ref_e100_{D // 1000}k", "cell": "mpra_legnet_ref", "idea": "published MPRA-LegNet, out-of-the-box recipe (e100, wd 0.1)", "orig_val": 0.0, "n_params": 1322465,
         "source": SOURCE, "train_config": {**base, "weight_decay": 0.1, "epochs": 100}},
        {"id": f"legnet_ref_tuned_{D // 1000}k", "cell": "mpra_legnet_ref", "idea": f"published MPRA-LegNet, per-size tuned recipe (e{ep}, wd {wd})", "orig_val": 0.0, "n_params": 1322465,
         "source": SOURCE, "train_config": {**base, "weight_decay": wd, "epochs": ep}},
    ]  # fmt: skip
    path = f"outputs/menus_nov2/PARITY_{D // 1000}k.json"
    if not DRY:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(entries, open(path, "w"), indent=1)
    return f"{R}/{path}"


def pa38_cells():
    out = []
    for D, cap in {10000: 40, 30000: 120, 100000: 240}.items():
        menu = parity_menu(D)
        pool = f"{NEST}/genomic_d{D}.npz"
        out += fb.cells(
            f"pa38_ref_{D // 1000}k",
            menu,
            pool,
            D,
            {"AL_PATIENCE": "9999", "AL_PRECISION": "fp16"},
            STD,
            shards=2,
            cap=cap,
            seeds=3,
        )
    return out


def loop_cell(name, dirname, rep, D, rounds, cap, budget, agent_cfg):
    c = lr.cell(name, dirname, rep, D, rounds, 2, agent_cfg)
    assert " --val_cache " in c["cmd"] and " --base_cap_min 25 " in c["cmd"]
    c["cmd"] = (
        c["cmd"]
        .replace(" --val_cache ", " --reference_legnet --val_cache ", 1)
        .replace(" --base_cap_min 25 ", f" --base_cap_min {cap} ", 1)
        .replace("{BUDGET}", str(budget))
    )
    c["env"] = {**c["env"], **H2}
    if D >= 100000:
        c["gpus"] = ["h100"]
    return c


def lv39_cells():
    out = []
    for D, (reps, rounds, cap, budget) in LV_SIZES.items():
        for r in range(1, reps + 1):
            out.append(
                loop_cell(
                    f"lv39_{D // 1000}k_r{r}",
                    f"genomic_{D // 1000}k_lv39",
                    r,
                    D,
                    rounds,
                    cap,
                    budget,
                    COORD_VARIANTS,
                )
            )
    for c in out:  # staged: the 30k arm first, 10k and 100k when the LLM account limit allows (the first launch hit it with 11 loop cells)
        if "_30k_" not in c["name"]:
            c["needs"] = c["needs"] + [f"{GATES}/lv39b_go"]
    return out


def rr39_cells():
    reps, cap, budget = RR_SIZE
    return [
        loop_cell(
            f"rr39_30k_r{r}",
            "genomic_30k_rr39",
            r,
            30000,
            15,
            cap,
            budget,
            RR | {"explore_frac": 0.4},
        )
        for r in range(1, reps + 1)
    ]


def ef39_cells():
    out = []
    reps, cap, budget = RR_SIZE
    for ef, tag in ((0.2, "20"), (0.6, "60"), (1.0, "100")):
        for r in range(1, reps + 1):
            c = loop_cell(
                f"ef39_{tag}_30k_r{r}",
                f"genomic_30k_ef39_{tag}",
                r,
                30000,
                15,
                cap,
                budget,
                {**RR, "explore_frac": ef},
            )
            c["needs"] = c["needs"] + [f"{GATES}/ef39_go"]
            out.append(c)
    return out


B1 = f"{R}/outputs/val_variants/val_B1.npz"
B1_BASELINE = (
    "BASELINE: the MPRA-LegNet (shown below), trained 20 times with its reference script on 30,000 sequences of this kind, is the reference architecture (tuned recipe: AdamW, learning rate 0.01, weight decay 0.2, "
    "500 epochs, batch 1024, one-cycle). This validation set mixes seven sequence families (genomic, mutated genomic, EvoAug, random at several GC levels, planted motifs, orthologs, oracle-guided designs), so validation r is "
    "lower than on genomic sequences alone and no reference number is quoted for it; what is scored is the ensemble of everything you train."
)


def vc39_cells():
    out = []
    reps, cap, budget = RR_SIZE
    for r in range(1, reps + 1):
        c = loop_cell(
            f"vc39_30k_r{r}",
            "genomic_30k_vc39",
            r,
            30000,
            15,
            cap,
            budget,
            RR | {"explore_frac": 0.4},
        )
        assert lr.V0C in c["cmd"]
        c["cmd"] = c["cmd"].replace(lr.V0C, B1)
        c["needs"] = c["needs"] + [f"{GATES}/vc39_go", "outputs/val_variants/val_B1.npz"]
        if not DRY:  # the launcher keeps an existing baseline_legnet.txt: the genomic bar would not apply to this validation set
            d = f"{R}/outputs/archsearch_reps_oct22/genomic_30k_vc39/rep{r}"
            os.makedirs(d, exist_ok=True)
            if not os.path.exists(f"{d}/baseline_legnet.txt"):
                open(f"{d}/baseline_legnet.txt", "w").write(B1_BASELINE)
        out.append(c)
    return out


def main():
    cells = []
    for tag, fn in (
        ("pa38", pa38_cells),
        ("lv39", lv39_cells),
        ("rr39", rr39_cells),
        ("ef39", ef39_cells),
        ("vc39", vc39_cells),
    ):
        if tag in ONLY:
            cells += fn()
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells at the end | registry", len(reg), "->", len(reg2))
        for c in new:
            print("  ", c["name"], "needs", [os.path.basename(n) for n in c["needs"]])
        if DRY:
            print(copy.deepcopy(new[0])["cmd"][:700])
            return
        os.makedirs(GATES, exist_ok=True)
        shutil.copy(path, path + ".bak_nov2b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered; release a gated group with: touch", f"{GATES}/<ef39|vc39>_go")


if __name__ == "__main__":
    main()
