"""Nov-9: re-run the Malinois rows of the 30k reference bracket, which never built.

Every Malinois entry registered by ``register_nov9m.py`` failed the harness's source gate with
``build: banned name: getattr`` -- the published ``Basset`` resolves its activation with
``getattr(nn, activation)``, and ``albench/arch_search/validate.py`` bans that name. The register
guard there missed it because it checked the source with a plain ``exec`` rather than with
``static_check``, which is the harness's actual gate; both the model and that guard are now fixed.

Only Malinois is re-run, and under the same fp16/aug environment and the same six recipes as the
other two architectures, so the three rows stay comparable at a common point. The DREAM-RNN and
DREAM-Attn rows of the original bracket are untouched and still running.

    python scripts/program_oct22/register_nov9n.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_nov9m as m  # noqa: E402

R, OUT, DRY = m.R, m.OUT, m.DRY
D, SEEDS, SHARDS = m.D, m.SEEDS, 2
TAG = f"refbankb_{D // 1000}k_genomic_can"
MENU_PATH = f"outputs/menus_nov9/REFBANKB_{D // 1000}k.json"
ARCHS = tuple(a for a in m.ARCHS if a[0] == "malinois")


def main():
    import torch

    from albench.arch_search.harness_standard import INPUT_LEN
    from albench.arch_search.validate import static_check

    assert len(ARCHS) == 1, ARCHS
    for tag, mod, n_params in ARCHS:
        bad = static_check(mod.SOURCE)
        assert bad is None, f"{tag}: the harness validator still rejects this source -- {bad}"
        ns = {}
        exec(compile(mod.SOURCE, f"<{tag}>", "exec"), ns)
        net = ns["Net"]()
        got = sum(p.numel() for p in net.parameters())
        assert got == n_params, f"{tag}: {got:,} parameters, menu claims {n_params:,}"
        net.eval()
        with torch.no_grad():
            y = net(torch.randn(2, 4, INPUT_LEN))
        assert y.shape == (2,) and torch.isfinite(y).all(), f"{tag}: output {tuple(y.shape)}"
    print(f"guard: malinois passes the harness validator and builds at {INPUT_LEN} bp")

    # the same entry shape and ids as the original bracket, so the two out_dirs read as one table
    entries = [e for e in build_entries() if e["cell"] == "malinois_ref"]
    assert len(entries) == len(m.RECIPES), len(entries)
    if not DRY:
        os.makedirs(os.path.join(R, os.path.dirname(MENU_PATH)), exist_ok=True)
        json.dump(entries, open(os.path.join(R, MENU_PATH), "w"), indent=1)

    odir = f"{OUT}/{TAG}"
    pool = f"{m.NEST}/genomic_d{D}.npz"
    cells = [
        dict(
            name=f"{TAG}_s{k}",
            gpus=["h100", "v100"],
            ext=[],
            out_dir=odir,
            env=dict(m.ENV),
            needs=[pool, MENU_PATH],
            cmd=(
                f"uv run --no-sync python scratchpad/menu_retrain.py "
                f"--menu {R}/{MENU_PATH} --reservoir_cache {R}/{pool} --D {D} "
                f"--out_dir {R}/{odir} --seeds {SEEDS} --shard {k}/{SHARDS} "
                f"--base_cap_min 240 --val_cache {m.VAL} --flank_ctx 20 --shift_max 10"
            ),
            done=dict(type="file", path=f"{odir}/COMPLETE_{k}of{SHARDS}"),
        )
        for k in range(SHARDS)
    ]
    for c in cells:
        assert " --shift_max 10" in c["cmd"] and " --flank_ctx 20" in c["cmd"], (
            "the 240-bp standard"
        )
        od = os.path.join(R, c["out_dir"])
        assert not os.path.isdir(od) or not os.listdir(od), f"{c['out_dir']} is not empty"
    print(f"guard: {len(cells)} cells, {len(entries)} entries x {SEEDS} seeds, menu at {MENU_PATH}")

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
            return
        shutil.copy(path, path + ".bak_nov9n")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


def build_entries():
    """The original bracket's entry list, rebuilt without writing its menu file."""
    out = []
    for tag, mod, n_params in m.ARCHS:
        for ep, wd, lr in m.RECIPES:
            out.append(
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
    return out


if __name__ == "__main__":
    main()
