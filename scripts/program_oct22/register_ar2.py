"""Register GPU worker cells for the ar2 (AutoResearch v2) searches in the dispatcher.

Each search has a workspace outputs/ar2/ws_<cell>; its orchestrator (a tmux session on the login node, started by scripts/ar2_launch.sh) writes training
specs into ws/queue/pending, and these worker cells (python -m ar2 worker --ws ...) claim and run them on a GPU. A worker exits when the queue has been empty for
--idle-exit-s; the dispatcher resubmits it until the workspace's DONE file exists, so a worker does not hold a GPU through long LLM calls but comes back when
there is work again. Cells go right after the pool31_ cells (ahead of the scaling and top-up grids) so a search is never starved by the grid.

  python scripts/program_oct22/register_ar2.py --cells genomic_10k:2,genomic_30k:3,random_30k:2,motif_30k:2 [--dry]
"""

import json
import os
import shutil
import sys

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
os.chdir(R)
arg = sys.argv[sys.argv.index("--cells") + 1] if "--cells" in sys.argv else "genomic_10k:2"
cells = [(c.split(":")[0], int(c.split(":")[1])) for c in arg.split(",")]
reg = json.load(open("scratchpad/dispatch_registry.json"))
have = {c["name"] for c in reg}
new = []
for cell, n in cells:
    ws = f"outputs/ar2/ws_{cell}"
    for k in range(n):
        name = f"ar2w_{cell}_{k}"
        if name in have:
            continue
        new.append(
            dict(
                name=name,
                gpus=["h100", "v100"],
                ext=[],
                out_dir=ws,
                env={"MENU_MAX_FWD_MS": "1000000000"},
                needs=[f"{ws}/config.yaml"],
                cmd=f"uv run --no-sync python -m ar2 worker --ws {R}/{ws} --idle-exit-s 900",
                done=dict(type="file", path=f"{ws}/DONE"),
            )
        )
i = max(
    k for k, c in enumerate(reg) if c["name"].startswith("pool31_")
)  # right after the new pool builds: ahead of the scaling and top-up grids
reg2 = reg[: i + 1] + new + reg[i + 1 :]
print("new worker cells", len(new), "registry", len(reg), "->", len(reg2))
if "--dry" not in sys.argv:
    shutil.copy("scratchpad/dispatch_registry.json", "scratchpad/dispatch_registry.json.bak_ar2")
    json.dump(reg2, open("scratchpad/dispatch_registry.json.tmp", "w"), indent=1)
    os.replace("scratchpad/dispatch_registry.json.tmp", "scratchpad/dispatch_registry.json")
    print("registered")
