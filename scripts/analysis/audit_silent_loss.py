"""Find runs marked COMPLETE that silently lost models.

menu_retrain writes COMPLETE_{k}of{n} when a shard exits, whether or not its entries trained.
A directory with COMPLETE markers AND a non-empty failed.jsonl has therefore been counted as
finished while contributing fewer models than its menu asked for. This lists them, worst first,
and groups the reasons so a systematic cause (slice OOM, a banned name, fp16 overflow) is visible.
"""

import collections
import glob
import json
import os
import re

ROOTS = ("outputs/menu_retrain_v0", "outputs/archsearch_reps_oct22")
rows = []
for root in ROOTS:
    for dirpath, dirnames, filenames in os.walk(root):
        comp = [f for f in filenames if f.startswith("COMPLETE_")]
        fj = os.path.join(dirpath, "failed.jsonl")
        if not os.path.exists(fj):
            continue
        reasons = collections.Counter()
        n_fail = 0
        for line in open(fj):
            line = line.strip()
            if not line:
                continue
            n_fail += 1
            try:
                why = json.loads(line).get("why", "?")
            except Exception:
                why = "unparseable"
            key = (
                re.sub(r"\d+", "N", why.split(":")[0] + ":" + why.split(":")[1][:40])
                if ":" in why
                else why[:40]
            )
            reasons[key.strip()] += 1
        if not n_fail:
            continue
        n_models = len(glob.glob(os.path.join(dirpath, "*_meta.json")))
        shards = set()
        for c in comp:
            m = re.match(r"COMPLETE_(\d+)of(\d+)$", c)
            if m:
                shards.add((int(m.group(1)), int(m.group(2))))
        total = max((n for _, n in shards), default=0)
        rows.append((n_fail, dirpath, n_models, len(shards), total, reasons))

rows.sort(reverse=True)
print(f"{'lost':>5}  {'models':>6}  {'shards':>7}  directory")
for n_fail, d, n_models, n_comp, total, reasons in rows:
    flag = "  <-- MARKED COMPLETE" if n_comp and n_comp >= total > 0 else ""
    print(f"{n_fail:>5}  {n_models:>6}  {n_comp:>3}/{total:<3}  {d}{flag}")
    for why, k in reasons.most_common(3):
        print(f"                        {k:>3} x {why}")
print(f"\n{len(rows)} directories with at least one lost model")
