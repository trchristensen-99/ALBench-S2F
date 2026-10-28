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
    ("300k nested baselines", ("mrr_nest_",)),
    ("noise x1 (10k/30k/100k)", ("ind30n1_",)),
    ("noise x2 (30k)", ("ind30n2_",)),
    ("validation phase 2", ("ind30v_",)),
    ("Zoonomia rebuild", ("pool30_zoo2", "un10_zoov2", "un30_zoov2")),
    ("nine seeds (30k)", ("ind30s_",)),
    ("seed ladder (10k/100k)", ("ind30l_",)),
    ("acq downstream (100k, K=8)", ("ind30q_",)),
    ("HepG2 / differential", ("ind30h_",)),
    ("oracle B", ("orB_",)),
    ("recipe sensitivity (epochs, patience)", ("ind30e_",)),
    ("validation V0c at 10k/100k", ("ind30w_",)),
    ("600k recipe retune", ("ind30t_",)),
    ("LLM: loop controls (plain agent, random pick)", ("ag30_p_", "ag30_x_")),
    ("LLM: guidance none", ("xd30_gn_",)),
    ("LLM: 300k searches", ("g300s_",)),
    ("LLM: agent variants", ("ac30_", "an30_", "ae30_")),
    ("LLM: explore grid", ("xe30_", "xd30_", "xe10_", "xd10_")),
    ("Oct 28: genomic ensembles with weights", ("ind30z_",)),
    ("Oct 28: patience 16 on pools r2/r3", ("ind30e_pat16_",)),
    (
        "Oct 28: random/motif 10k+100k r1-r3",
        ("un10_random_", "un10_motif_ct_enriched_", "un100_random_", "un100_motif_ct_enriched_"),
    ),
    ("Oct 28: shared core 300k", ("pool100_sc300", "mrr_nest_motif_shared_core")),
    ("Oct 28 LLM: separate coder, other reservoirs", ("ag30_p_rnd", "ag30_p_mot", "ag30_p_gen10")),
    ("Oct 28 LLM: coordinator, evoaug/evo2 10k", ("an30_e10", "an30_v10")),
    ("Oct 29: patience 24 and 32", ("ind30e_pat24_", "ind30e_pat32_")),
    ("Oct 29: motif-run menu MM + held-out reservoirs", ("as_MM_", "as_MG_evo", "as_MR_evo")),
    ("Oct 29: width-scaled configs (generator + retrains)", ("scl30_",)),
    ("Oct 29: menu v1 validation retrains", ("ind30m_v1_",)),
    ("Oct 28 night: new pools (GC sweep, 10% mutagenesis)", ("pool31_",)),
    (
        "Oct 28 night: scaling 10k (menu v1, 8 strategies)",
        (
            "sc31_random_10k",
            "sc31_genomic_10k",
            "sc31_motif_ct_enriched_10k",
            "sc31_evoaug_paper2023_10k",
            "sc31_zoonomia_orthologs_10k",
            "sc31_motif_shared_core_10k",
            "sc31_random_gc_sweep_10k",
            "sc31_mutagenesis_mr10_10k",
        ),
    ),
    (
        "Oct 28 night: scaling 30k",
        tuple(
            f"sc31_{s}_30k"
            for s in (
                "genomic",
                "random",
                "motif_ct_enriched",
                "evoaug_paper2023",
                "zoonomia_orthologs",
                "motif_shared_core",
                "random_gc_sweep",
                "mutagenesis_mr10",
            )
        ),
    ),
    (
        "Oct 28 night: scaling 100k (12 configs)",
        tuple(
            f"sc31_{s}_100k"
            for s in (
                "genomic",
                "random",
                "motif_ct_enriched",
                "evoaug_paper2023",
                "zoonomia_orthologs",
                "motif_shared_core",
                "random_gc_sweep",
                "mutagenesis_mr10",
            )
        ),
    ),
    ("Oct 28 night: 30k genomic + 70k arms (reservoir, mix, acquisition)", ("al31_",)),
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
