#!/usr/bin/env python3
"""Check Borzoi SNV and NTv3 K562 anomalies in detail."""

import json
from pathlib import Path


def fmt(x):
    return f"{x:.4f}" if x is not None else "None"


# Borzoi SNV check
print("=== Borzoi SNV ===")
for cell in ["hepg2", "sknsh", "k562"]:
    for pat in [f"borzoi_{cell}_cached", f"borzoi_{cell}_3seeds"]:
        p = Path("outputs") / pat
        if not p.exists():
            continue
        for f in sorted(p.rglob("result.json"))[:1]:
            r = json.loads(f.read_text())
            tm = r.get("test_metrics", {})
            snv_a = tm.get("snv_abs", {})
            snv_d = tm.get("snv_delta", {})
            print(
                f"  {pat}: snv_abs r={fmt(snv_a.get('pearson_r'))} "
                f"n={snv_a.get('n', '?')} | "
                f"snv_delta r={fmt(snv_d.get('pearson_r'))} "
                f"n={snv_d.get('n', '?')}"
            )

# NTv3 K562 cached check
print("\n=== NTv3 K562 cached ===")
p = Path("outputs/ntv3_post_k562_cached")
if p.exists():
    json_files = sorted(p.rglob("*.json"))
    result_files = sorted(p.rglob("result.json"))
    print(f"  {len(json_files)} json files, {len(result_files)} result.json")
    for f in result_files[:3]:
        r = json.loads(f.read_text())
        tm = r.get("test_metrics", {})
        id_r = tm.get("in_dist", tm.get("in_distribution", {})).get("pearson_r")
        ood = tm.get("ood", {}).get("pearson_r")
        print(f"  {f.relative_to(p)}: id={fmt(id_r)} ood={fmt(ood)}")
else:
    print("  NOT FOUND")

# Enformer K562 OOD check (0.196 seems low)
print("\n=== Enformer K562 S1 (OOD check) ===")
p = Path("outputs/enformer_k562_3seeds")
if p.exists():
    for f in sorted(p.rglob("result.json"))[:3]:
        r = json.loads(f.read_text())
        tm = r.get("test_metrics", {})
        id_r = tm.get("in_dist", tm.get("in_distribution", {})).get("pearson_r")
        ood = tm.get("ood", {}).get("pearson_r")
        print(f"  {f.relative_to(p)}: id={fmt(id_r)} ood={fmt(ood)}")

# AG S1 HepG2/SKNSH SNV check (0.77 seems low compared to K562 0.89)
print("\n=== AG S1 SNV comparison ===")
for cell_pat in [
    ("k562", "ag_hashfrag_oracle_cached"),
    ("hepg2", "ag_hashfrag_hepg2_cached"),
    ("sknsh", "ag_hashfrag_sknsh_cached"),
]:
    cell, pat = cell_pat
    p = Path("outputs") / pat
    if not p.exists():
        continue
    for f in sorted(p.rglob("test_metrics.json"))[:1]:
        r = json.loads(f.read_text())
        tm = r.get("test_metrics", r)
        snv_a = tm.get("snv_abs", {})
        snv_d = tm.get("snv_delta", {})
        id_r = tm.get("in_distribution", tm.get("in_dist", {})).get("pearson_r")
        print(
            f"  AG S1 {cell}: id={fmt(id_r)} "
            f"snv_abs={fmt(snv_a.get('pearson_r'))} (n={snv_a.get('n', '?')}) "
            f"snv_delta={fmt(snv_d.get('pearson_r'))} (n={snv_d.get('n', '?')})"
        )
