"""Nov-6: three attempts to make the search loop find something better than a LegNet derivative, all on Opus 5.5.

Every top model the loop has ever produced is a LegNet derivative, and its best single model still sits below the bank's mean seed. Two prompt-level causes are testable, and switching the proposer to Opus is the
third thing asked for. Because the existing baseline `ar41` ran on Sonnet, an Opus baseline is included: without it a gain could be the model rather than the change.

  op49_30k_r<k>   ar41 settings, Opus 5.5                           the baseline the other two are read against
  pe49_30k_r<k>   + the proposer sees the validation stack          it has only ever optimised for the best SINGLE model; the
                                                                    coordinator saw the stack but the proposer never did
  xr49_30k_r<k>   + further reference architectures in the prompt   it is shown only MPRA-LegNet and copies it; C219 (the
                                                                    collaborator's ShuffleNet-style design, which beats LegNet
                                                                    in our hands) and the best independent design join it

All three otherwise match `ar41`: architecture-only search, every model on the tuned 300-epoch recipe, patience off, 15 rounds x 2, three replicate pools. Multi-seed selection is deliberately NOT here: it changes
the training path rather than the prompt, and `reg48_winners_30k` measures how much selection noise there is to remove before that code is written.

    python scripts/program_oct22/register_nov6e.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov2b as nb  # noqa: E402

R, DRY = nb.R, nb.DRY
OPUS = "claude-opus-5-5"
TUNED_30K = "adamw,0.01,0.2,1024,300,onecycle"
BASE_ENV = {"AL_RC_MODE": "aug", "AL_RECIPE_MODE": "tuned", "AL_TUNED_RECIPE": TUNED_30K, "AL_PATIENCE": "9999", "LLM_MODEL": OPUS}  # fmt: skip
REFS = f"{R}/outputs/menus_nov6/EXTRA_REFERENCES.json"


def write_refs():
    """C219 from the collaborator re-implementations, and the best independent design from menu v3."""
    out = []
    pi = [
        e for e in json.load(open(f"{R}/outputs/menus_nov3/PI41_mid_30k.json")) if "c219" in e["id"]
    ]
    if pi:
        out.append(
            {
                "name": "C219, a ShuffleNet-style design (not LegNet-derived)",
                "val_r": 0.9083,
                "source": pi[0]["source"],
            }
        )
    try:
        menu = json.load(open(f"{R}/outputs/menus_nov4/MENU_V3_30k.json"))
        ind = [e for e in menu if e.get("kind") == "independent"]
        if ind:
            out.append(
                {
                    "name": f"An independent design found by an earlier run: {ind[0]['idea'][:60]}",
                    "val_r": float(ind[0].get("orig_val") or 0.0),
                    "source": ind[0]["source"],
                }
            )
    except Exception as ex:  # noqa: BLE001
        print("  note: menu v3 unavailable for the second reference:", type(ex).__name__)
    assert out, "no extra references could be assembled"
    if not DRY:
        os.makedirs(os.path.dirname(REFS), exist_ok=True)
        json.dump(out, open(REFS, "w"), indent=1)
    for r in out:
        print(f"    reference: {r['name'][:66]} (val {r['val_r']:.4f}, {len(r['source'])} chars)")
    return out


def all_cells():
    reps, cap, budget = nb.RR_SIZE
    arms = (
        ("op49", {}),
        ("pe49", {"AL_PROPOSER_ENSEMBLE": "1"}),
        ("xr49", {"AL_EXTRA_REFERENCES": REFS}),
    )
    out = []
    for tag, extra in arms:
        for r in range(1, reps + 1):
            c = nb.loop_cell(
                f"{tag}_30k_r{r}", f"genomic_30k_{tag}", r, 30000, 15, cap, budget,
                {**nb.RR, "manager": "schedule_ens", "explore_frac": 0.4, "recipe_source": "harness"},
            )  # fmt: skip
            c["env"] = {**c["env"], **BASE_ENV, **extra}
            c["gpus"] = ["v100"]
            out.append(c)
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_refs()
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in all_cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            for c in new[:1] + new[3:4] + new[6:7]:
                print(
                    " ",
                    c["name"],
                    {
                        k: v
                        for k, v in c["env"].items()
                        if k.startswith(("AL_PROP", "AL_EXTRA", "LLM"))
                    },
                )
            return
        shutil.copy(path, path + ".bak_nov6e")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
