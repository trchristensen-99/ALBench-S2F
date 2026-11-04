"""Nov-4: the two Volterra (second-order convolution) architectures of the collaborator's list, C209 and C218, in float32.

register_nov3f.py's `mrr_pi41_*` cells trained the five architectures for 300 epochs under harness H3 in fp16; C209 and C218 diverged (non-finite loss at epoch 31-47) in every recipe, while C219, C213 and C210 trained
fine (C219 reaches 0.877 genomic test r). The Volterra term is a quadratic form, the likely reason for the fp16 overflow. This reruns just those two with AL_PRECISION=fp32 under the same four recipes (tuned, mid, own Muon, own AdamW; 300 epochs; 2 seeds).

  mrr_pi41f_<recipe>_30k_s0 (C209), _s1 (C218)       `mrr_` keeps them off the 12 GB MIG slices (fp32, 11M parameters)

    python scripts/program_oct22/register_nov4b.py [--dry]
"""

import fcntl
import json
import os
import shutil
import sys

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import register_fx35b as fb  # noqa: E402
import register_nov2b as nb  # noqa: E402
import register_nov3f as r3f  # noqa: E402

R, NEST, DRY = nb.R, nb.NEST, nb.DRY
D = 30000
FFT_OLD = "torch.fft.rfft(h - h.mean(dim=2, keepdim=True), dim=2).abs()"
FFT_NEW = "torch.fft.rfft((h - h.mean(dim=2, keepdim=True)).float(), dim=2).abs().to(h.dtype)"


def cells():
    base = [
        e
        for e in json.load(open(f"{R}/outputs/menus_oct31/PI.json"))
        if e["id"] in ("pi_c209_adamw", "pi_c218_adamw")
    ]
    assert [e["id"] for e in base] == ["pi_c209_adamw", "pi_c218_adamw"], [e["id"] for e in base]
    out = []
    for name, recipe in r3f.RECIPES.items():
        ents = [
            {
                **{k: v for k, v in e.items() if k != "train_config"},
                "id": e["id"].replace("_adamw", "") + f"_{name}",
                "train_config": dict(recipe),
                "source": e["source"].replace(FFT_OLD, FFT_NEW),
            }
            for e in base
        ]
        path = f"{R}/outputs/menus_nov3/PI41F_{name}_30k.json"
        if not DRY:
            json.dump(ents, open(path, "w"), indent=1)
        out += fb.cells(
            f"mrr_pi41f_{name}_30k", path, f"{NEST}/genomic_d{D}.npz", D,
            {"AL_PATIENCE": "9999", "AL_PRECISION": "fp32", **r3f.H3}, nb.STD, shards=2, cap=240, seeds=2,
        )  # fmt: skip
    return out


def main():
    path = "scratchpad/dispatch_registry.json"
    with open("scratchpad/.registry.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = json.load(open(path))
        have = {c["name"] for c in reg}
        new = [c for c in cells() if c["name"] not in have]
        reg2 = reg + new
        assert len({c["name"] for c in reg2}) == len(reg2)
        print("add", len(new), "cells | registry", len(reg), "->", len(reg2))
        if DRY:
            return
        shutil.copy(path, path + ".bak_nov4b")
        json.dump(reg2, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("registered")


if __name__ == "__main__":
    main()
