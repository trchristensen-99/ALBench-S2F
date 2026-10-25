"""Compact progress report for the dispatcher program: GPUs in use, per-group cell progress, alerts. Run on the HPC login node (read-only, light).

cd /grid/wsbs/home_norepl/christen/ALBench-S2F && .venv/bin/python scripts/program_oct22/status_oct26.py
"""

import collections
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad")
import dispatcher as d  # noqa: E402

os.chdir(d.R)
GROUPS = [
    ("acq scoring", ("ind30a_",)),
    ("1M pool builds", ("pool1m_",)),
    ("anchor 600k", ("a600_",)),
    ("anchor 1M", ("a1m_",)),
    ("noise x1 (10k/30k/100k)", ("ind30n1_",)),
    ("noise x2 (30k)", ("ind30n2_",)),
    ("validation phase 2", ("ind30v_",)),
    ("Zoonomia rebuild", ("pool30_zoo2", "un10_zoov2", "un30_zoov2")),
    ("nine seeds (30k)", ("ind30s_",)),
    ("seed ladder (10k/100k)", ("ind30l_",)),
    ("acq downstream (100k, K=8)", ("ind30q_",)),
    ("HepG2 / differential", ("ind30h_",)),
    ("oracle B", ("orB_",)),
    ("LLM: 300k searches", ("g300s_",)),
    ("LLM: agent variants", ("ac30_", "an30_", "ae30_")),
    ("LLM: explore grid", ("xe30_", "xd30_", "xe10_", "xd10_")),
]
reg = json.load(open(d.REG))
state = json.load(open(d.STATE_F)) if d.STATE_F.exists() else {}
mine = d.my_jobs()
free = d.free_gpus()
byname = collections.defaultdict(list)
for j in mine:
    byname[j["name"]].append(j)
print(
    time.strftime("%m-%d %H:%M"),
    f"| free GPUs in cluster: H100 {free['h100']}, V100 {free['v100']}",
)
cnt = collections.Counter((j["gpu"], j["st"]) for j in mine)
print(
    "mine:",
    ", ".join(f"{g} {st}={n}" for (g, st), n in sorted(cnt.items())),
    "| by QOS running:",
    dict(collections.Counter(j["qos"] for j in mine if j["st"] == "R")),
)
print(f"{'group':30s} {'done':>9s} {'run':>4s} {'pend':>4s} {'waiting':>7s}")
for label, pfx in GROUPS:
    cells = [c for c in reg if c["name"].startswith(pfx)]
    if label == "validation phase 2":
        cells = [c for c in cells if "_o1_" in c["name"]]
    if not cells:
        continue
    done = sum(1 for c in cells if d.is_done(c))
    run = pend = 0
    for c in cells:
        for j in byname.get("d_" + c["name"], []):
            run += j["st"] == "R"
            pend += j["st"] == "PD"
    print(
        f"{label:30s} {done:4d}/{len(cells):<4d} {run:4d} {pend:4d} {len(cells) - done - run - pend:7d}"
    )
gl = state.get("_global", {})
if time.time() < gl.get("llm_cooldown_until", 0) or gl.get("probe_needed"):
    print(
        "LLM submissions paused until",
        time.strftime("%m-%d %H:%M", time.localtime(gl.get("llm_cooldown_until", 0))),
    )
# alerts: failures logged by menu_retrain, stalls and failed submissions in the last 3 h of the dispatcher log
alerts = []
cut = time.time() - 3 * 3600
for f in glob.glob("outputs/menu_retrain_v0/*/failed.jsonl"):
    if os.path.getmtime(f) > cut:
        alerts.append(f"failures logged: {f.split('/')[-2]} ({sum(1 for _ in open(f))} lines)")
for line in open("outputs/dispatch/dispatch.log").read().splitlines()[-4000:]:
    m = re.match(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\]\s+(.*)", line)
    if m and ("STALL" in line or "FAILED" in line):
        t = time.mktime(time.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S"))
        if t > cut:
            alerts.append(line[:160])
print("alerts:", "none" if not alerts else "")
for a in alerts[:12]:
    print("  ", a)
