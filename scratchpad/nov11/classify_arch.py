#!/usr/bin/env python3
"""Classify every loop proposal by what its GENERATED CODE contains, not by its prose.

The claim "LegNet-style architectures dominate the loop" was measured by grepping the
proposal text, which says what the model-writer *intended*. This reads the `source`
field -- the code that was actually compiled and trained -- and also records which
reference architecture each arm was handed, because an arm launched with
--reference_legnet is being shown LegNet as the bar to beat.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

ROOT = "/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/archsearch_reps_oct22"

# Structural motifs, matched against the generated source. Deliberately about
# MECHANISM (does it carry state over positions? does it mix channels in groups?)
# rather than about lineage names, which is the thing we cannot defend.
MOTIFS = {
    "recurrent":    r"nn\.(GRU|LSTM|RNN)\b",
    "attention":    r"MultiheadAttention|scaled_dot_product_attention|TransformerEncoder|class\s+\w*Attention",
    "grouped_conv": r"groups\s*=\s*(?!1\b)\w+",
    "squeeze_exc":  r"\bSE\b|Squeeze|se_|_se\b",
    "dilated":      r"dilation\s*=\s*(?!1\b)\w+",
    "channel_shuf": r"channel_shuffle|ShuffleNet|shuffle",
    "depthwise":    r"depthwise|dw_conv|DWConv",
    "global_pool":  r"AdaptiveAvgPool|mean\(\s*(dim|axis)\s*=\s*(-1|2)",
    "inv_residual": r"InvertedResidual|expand_ratio|EffBlock|MBConv",
}


def arm_reference(arm_dir: str) -> str:
    """Which architecture was this arm shown as its reference?"""
    for rep in sorted(os.listdir(arm_dir)):
        cfg = os.path.join(arm_dir, rep, "config.json")
        if not os.path.exists(cfg):
            continue
        try:
            argv = " ".join(json.load(open(cfg)).get("argv", []))
        except Exception:
            continue
        if "--reference_legnet" in argv:
            return "legnet"
        m = re.search(r"--baseline_file\s+(\S+)", argv)
        if m:
            return os.path.basename(m.group(1)).replace("_BASELINE.txt", "").lower()
        return "none"
    return "unknown"


def main() -> None:
    per_arm = {}
    rows = []
    for arm in sorted(os.listdir(ROOT)):
        arm_dir = os.path.join(ROOT, arm)
        if not os.path.isdir(arm_dir):
            continue
        ref = arm_reference(arm_dir)
        per_arm[arm] = ref
        for rep in sorted(os.listdir(arm_dir)):
            rep_dir = os.path.join(arm_dir, rep)
            if not os.path.isdir(rep_dir):
                continue
            for fn in sorted(os.listdir(rep_dir)):
                if not fn.endswith("_meta.json") or "_arch_" not in fn:
                    continue
                try:
                    m = json.load(open(os.path.join(rep_dir, fn)))
                except Exception:
                    continue
                src = m.get("source") or ""
                if not src:
                    continue
                hits = tuple(sorted(k for k, pat in MOTIFS.items()
                                    if re.search(pat, src, re.I)))
                rows.append({
                    "arm": arm, "ref": ref, "rep": rep, "model_id": m.get("model_id"),
                    "val": m.get("val_pearson"), "test": m.get("test_oracle_pearson"),
                    "n_params": m.get("n_params"), "motifs": hits,
                })
    out = "/grid/wsbs/home_norepl/christen/ALBench-S2F/scratchpad/nov11/arch_motifs.json"
    with open(out, "w") as fh:
        json.dump([{**r, "motifs": list(r["motifs"])} for r in rows], fh)

    print("proposals with code: %d over %d arms" % (len(rows), len(per_arm)))
    byref = defaultdict(int)
    for a, r in per_arm.items():
        byref[r] += 1
    print("\n=== what reference each ARM was handed ===")
    for r, n in sorted(byref.items(), key=lambda kv: -kv[1]):
        print("  %-12s %3d arms" % (r, n))

    scored = [r for r in rows if r["val"] is not None]
    scored.sort(key=lambda r: -r["val"])
    print("\n=== motif frequency: all proposals vs the top 100 by val ===")
    n_all = len(scored)
    top = scored[:100]
    print("  %-14s %8s %8s" % ("motif", "all%", "top100%"))
    for k in MOTIFS:
        a = 100.0 * sum(k in r["motifs"] for r in scored) / max(n_all, 1)
        t = 100.0 * sum(k in r["motifs"] for r in top) / max(len(top), 1)
        print("  %-14s %7.1f%% %7.1f%%" % (k, a, t))

    print("\n=== best proposal per reference class ===")
    best = {}
    for r in scored:
        if r["ref"] not in best:
            best[r["ref"]] = r
    for ref, r in sorted(best.items(), key=lambda kv: -(kv[1]["val"] or 0)):
        print("  ref=%-10s val %.4f test %.4f  %s/%s  motifs=%s"
              % (ref, r["val"], r["test"] or float("nan"), r["arm"], r["model_id"],
                 ",".join(r["motifs"]) or "-"))
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
PY
