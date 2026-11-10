"""Nov-9: tune the three new published references at 30k, so the reference bank spans four architecture families.

The bank is MPRA-LegNet, DREAM-RNN, DREAM-Attn and Malinois: a convolutional tower, a bidirectional LSTM,
a Conformer stack and a Basset CNN. DREAM-CNN is deliberately absent -- it is the Prix Fixe model built on
Autosome.org's core block, the same group as MPRA-LegNet, so it would be close to a duplicate of a member
the bank already has.

Each reference gets its OWN recipe at each size rather than LegNet's. That reverses the rule proposals are
held to, and the reason is the Nov-3 bake-off: independent designs gained +0.03 to +0.05 when given the
tuned recipe instead of their own, which says the recipe is worth more than most architectural differences.
Enforcing one recipe is therefore the right control for "is this architectural change an improvement" and the
wrong protocol for "does this architecture family contribute" -- it evaluates every non-LegNet family at
LegNet's optimum, which is a mechanism for the understated non-LegNet members we measured today (ar41's
independent half stacks to 0.8673 against its LegNet-derived 0.8858 at 30k, and rr40's to 0.8946 against
0.9331 at 100k).

Six recipes per architecture, a coordinate bracket around the 30k entry of the per-size table (e500, wd 0.2),
with that entry shared so the architectures are comparable at a common point:

    (500, 0.2)  the table entry        (300, 0.2)  fewer epochs      (800, 0.2)  more epochs
    (500, 0.05) less decay            (500, 0.4)  more decay        (500, 0.2, lr 0.003)  lower peak LR

The weight-decay points span LegNet's own measured ladder (0.4 / 0.2 / 0.1 / 0.05 at 10k / 30k / 100k / 300k),
so each architecture can land anywhere on it. The learning-rate probe is there for DREAM-Attn specifically:
attention stacks are the likeliest of the four to want a lower peak LR, and finding that out after spending
the 100k budget would be the expensive order to do this in.

30k only, on purpose. These three architectures have never been trained in this harness, so the measurement
comes before the budget: ~19 GPU-hours here against ~76 for 10k, 30k and 100k together. If the bracket looks
sane, 10k and 100k follow with the winning recipes and more seeds.

    python scripts/program_oct22/register_nov9m.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_ln100 as ln  # noqa: E402

sys.path.insert(0, ln.R)
from albench.arch_search import dream_attn_ref, dream_rnn_ref, malinois_ref  # noqa: E402

R, OUT, DRY = ln.R, ln.OUT, ln.DRY
D = 30_000
SEEDS = 3
SHARDS = 6
TAG = f"refbank_{D // 1000}k_genomic_can"
MENU_PATH = f"outputs/menus_nov9/REFBANK_{D // 1000}k.json"
VAL = "/grid/koo/home/shared/albench/data/val_V0c.npz"
NEST = "outputs/grid_pools/k562_nested"
ENV = {
    "AL_PATIENCE": "9999",
    "MENU_MAX_FWD_MS": "1000000000",
    "AL_PRECISION": "fp16",
    "AL_RC_MODE": "aug",
}
# (architecture tag, module, measured parameter count at the 240-bp standard)
ARCHS = (
    ("dream_rnn", dream_rnn_ref, 4_199_553),
    ("dream_attn", dream_attn_ref, 4_620_289),
    ("malinois", malinois_ref, 2_971_901),
)
# (epochs, weight decay, peak lr); the first is the 30k entry of the per-size table
RECIPES = (
    (500, 0.2, 0.01),
    (300, 0.2, 0.01),
    (800, 0.2, 0.01),
    (500, 0.05, 0.01),
    (500, 0.4, 0.01),
    (500, 0.2, 0.003),
)


def write_menu() -> str:
    entries = []
    for tag, mod, n_params in ARCHS:
        for ep, wd, lr in RECIPES:
            entries.append(
                {
                    "id": f"{tag}_e{ep}_wd{str(wd).replace('.', '')}_lr{str(lr).replace('.', '')}",
                    "cell": f"{tag}_ref",
                    "idea": f"published {tag} at epochs {ep}, weight decay {wd}, peak lr {lr}",
                    "orig_val": 0.0,
                    "n_params": n_params,
                    "source": mod.SOURCE,
                    "train_config": {
                        "optimizer": "adamw",
                        "lr": lr,
                        "batch_size": 1024,
                        "lr_schedule": "onecycle",
                        "weight_decay": wd,
                        "epochs": ep,
                    },
                }
            )
    assert len({e["id"] for e in entries}) == len(entries), "two recipes collapsed to one id"
    assert len(entries) == len(ARCHS) * len(RECIPES), len(entries)
    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(MENU_PATH)), exist_ok=True)
        json.dump(entries, open(os.path.join(R, MENU_PATH), "w"), indent=1)
    return MENU_PATH


def all_cells():
    pool = f"{NEST}/genomic_d{D}.npz"
    odir = f"{OUT}/{TAG}"
    return [
        dict(
            name=f"{TAG}_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=dict(ENV),
            needs=[pool, MENU_PATH],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{MENU_PATH} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{SHARDS} "
                f"--base_cap_min 240 --val_cache {VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{SHARDS}"),
        )
        for k in range(SHARDS)
    ]


def main():
    # the architectures must pass the HARNESS's own gate before any of this is registered, not merely
    # import: the first run of this script used a plain exec, which accepted a Malinois that the
    # harness then refused for 18 models in a row with "build: banned name: getattr"
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    for tag, mod, n_params in ARCHS:
        bad = static_check(mod.SOURCE)
        assert bad is None, f"{tag}: the harness validator rejects this source -- {bad}"
        ns = {}
        exec(compile(mod.SOURCE, f"<{tag}>", "exec"), ns)
        net = ns["Net"]()
        got = sum(p.numel() for p in net.parameters())
        assert got == n_params, f"{tag}: {got:,} parameters, menu claims {n_params:,}"
        net.eval()
        with torch.no_grad():
            y = net(torch.randn(2, 4, INPUT_LEN))
        assert y.shape == (2,) and torch.isfinite(y).all(), f"{tag}: output {tuple(y.shape)}"
    print(
        f"guard: {len(ARCHS)} architectures build at {INPUT_LEN} bp with the parameter counts the menu states"
    )

    menu = write_menu()
    cells = all_cells()
    assert os.path.exists(os.path.join(R, f"{NEST}/genomic_d{D}.npz")), "pool missing"
    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(
        f"guard: {len(cells)} cells, {len(ARCHS) * len(RECIPES)} entries x {SEEDS} seeds, menu at {menu}"
    )

    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print(f"add {len(new)} cells | registry {len(reg)} -> {len(reg2)}")
        if DRY:
            print([c["name"] for c in new])
            print()
            print(new[0]["cmd"])
            return
        shutil.copy(path, path + ".bak_nov9m")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
