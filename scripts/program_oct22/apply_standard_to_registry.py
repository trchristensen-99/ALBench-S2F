"""One-off, run before the standard defaults take effect for new directories: every registered cell that has not finished and calls the retraining script or the AutoResearch launcher without any context flag gets an
explicit --no_context, so the cells that were designed as no-context experiments (the base arms of the sweeps, older scaling cells still waiting for inputs) keep doing what they were registered to do.

    python scripts/program_oct22/apply_standard_to_registry.py [--dry]
"""

import collections
import fcntl
import json
import os
import shutil
import sys

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
os.chdir(R)
DRY = "--dry" in sys.argv
path = "scratchpad/dispatch_registry.json"


def finished(c):
    d = c.get("done", {})
    if d.get("type") == "file":
        p = d["path"]
        return os.path.exists(p if os.path.isabs(p) else os.path.join(R, p))
    return False


with open("scratchpad/.registry.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    reg = json.load(open(path))
    n, by = 0, collections.Counter()
    for c in reg:
        cmd = c.get("cmd", "")
        if (
            ("menu_retrain.py" in cmd or "arch_autoresearch.py" in cmd)
            and not any(f in cmd for f in ("--flank_ctx", "--shift_max", "--no_context"))
            and not finished(c)
        ):
            c["cmd"] = cmd + " --no_context"
            n += 1
            by[c["name"].split("_")[0]] += 1
    print(
        "added --no_context to",
        n,
        "unfinished cells without context flags:",
        dict(by.most_common(12)),
    )
    if not DRY:
        shutil.copy(path, path + ".bak_std")
        json.dump(reg, open(path + ".tmp", "w"), indent=1)
        os.replace(path + ".tmp", path)
        print("written")
