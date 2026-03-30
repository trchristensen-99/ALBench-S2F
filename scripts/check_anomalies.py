#!/usr/bin/env python3
"""Check known anomalies in bar/scatter plot results."""

import json
from pathlib import Path

checks = {
    "NTv3 S1 K562": "outputs/ntv3_post_k562_cached",
    "Enformer S1 K562": "outputs/enformer_k562_3seeds",
    "AG S2 OLD hepg2": "outputs/ag_hepg2_stage2",
    "AG S2 NEW hepg2": "outputs/ag_all_folds_hepg2_s2_from_s1",
    "AG S2 OLD sknsh": "outputs/ag_sknsh_stage2",
    "AG S2 NEW sknsh": "outputs/ag_all_folds_sknsh_s2_from_s1",
    "NTv3 S2 K562 broken": "outputs/ntv3_k562_stage2_final",
    "NTv3 S2 K562 sweep": "outputs/ntv3_post_k562_stage2",
    "Borzoi S1 hepg2": "outputs/borzoi_hepg2_cached",
    "Borzoi S1 sknsh": "outputs/borzoi_sknsh_cached",
    "AG S1 hepg2": "outputs/ag_hashfrag_hepg2_cached",
    "AG S1 sknsh": "outputs/ag_hashfrag_sknsh_cached",
    "AG fold1 S2 hepg2": "outputs/ag_fold_1_hepg2_s2_from_s1",
    "AG fold1 S2 sknsh": "outputs/ag_fold_1_sknsh_s2_from_s1",
    "Enformer S2 hepg2": "outputs/enformer_hepg2_stage2",
    "Enformer S2 sknsh": "outputs/enformer_sknsh_stage2",
}


def fmt(x):
    return f"{x:.4f}" if x is not None else "None"


for label, dpath in checks.items():
    p = Path(dpath)
    if not p.exists():
        print(f"{label}: DIR NOT FOUND")
        continue
    found = False
    for f in sorted(p.rglob("test_metrics.json"))[:2]:
        found = True
        r = json.loads(f.read_text())
        tm = r.get("test_metrics", r)
        id_r = tm.get("in_distribution", tm.get("in_dist", {})).get("pearson_r")
        snv_a = tm.get("snv_abs", {}).get("pearson_r")
        snv_d = tm.get("snv_delta", {}).get("pearson_r")
        ood = tm.get("ood", {}).get("pearson_r")
        n_snv = tm.get("snv_abs", {}).get("n", "?")
        rel = str(f.relative_to(Path("outputs")))
        print(
            f"{label}: id={fmt(id_r)} snv_a={fmt(snv_a)}(n={n_snv})"
            f" snv_d={fmt(snv_d)} ood={fmt(ood)} [{rel}]"
        )
    if not found:
        print(f"{label}: NO test_metrics.json")
