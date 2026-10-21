"""Build the cross-D context blocks for the free-architecture loop (the 'ladder' hand-off from a smaller dataset size to a larger one).

Arms (text files, identical for every rep of an arm so the only variance is the LLM + training seeds):
  T  top-k architectures by VALIDATION Pearson from the smaller-D AutoResearch instances (idea, val r, size, training recipe)
  J  T + a rule-based summary of how those runs progressed (best-so-far by round block, architecture-family mix early vs late, recipe drift)
  X  like T but from an even smaller D (10k; ranked by the retrained menu's validation) -- the control for 'context from a size whose winners do
     not transfer'
Only validation numbers are used: no test-set information reaches the proposer.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import numpy as np

FAMILIES = {
    "squeeze-excite / inverted-residual conv towers": r"squeeze|excitation|\bse\b|inverted|mbconv|legnet|effblock|residual|resnet|dense",
    "dilated / multi-scale convolution": r"dilat|multi-?scale|inception|pyramid|wavenet",
    "attention / transformer hybrids": r"attention|transformer|self-att|cbam|multi-?head",
    "U-Net / encoder-decoder": r"u-?net|encoder-decoder|skip",
    "gated / state-space / recurrent": r"gated|gating|mamba|state.?space|ssm|lstm|gru|recurrent|hyena",
    "explicit motif / grammar models": r"motif|grammar|spacing|pairwise|interaction|pwm|thermodynamic",
}
FOOT = ("Larger datasets may favour different depth, capacity or regularization than smaller ones, and which architectures rank highest can change "
        "with dataset size; treat this as evidence about what worked, not as a template.")


def family(idea: str) -> str:
    for name, pat in FAMILIES.items():
        if re.search(pat, idea or "", re.I):
            return name
    return "other"


def recipe(tc: dict) -> str:
    return "%s lr %.0e wd %.0e %s bs %s" % (tc.get("optimizer", "?"), float(tc.get("lr", 0)), float(tc.get("weight_decay", 0)),
                                           tc.get("lr_schedule", "?"), tc.get("batch_size", "?"))


def load_cells(dirs):
    rows = []
    for ci, d in enumerate(dirs):
        for f in glob.glob(d.rstrip("/") + "/r*_arch_*_meta.json"):
            try:
                m = json.load(open(f))
            except Exception:
                continue
            if m.get("val_pearson") is None or not m.get("idea") or not m.get("train_config"):
                continue
            rows.append(dict(cell=ci, round=int(m["round"]), val=float(m["val_pearson"]), idea=m["idea"], n=m.get("n_params") or 0,
                             tc=m["train_config"]))
    return rows


def top_block(rows, k, title):
    seen, lines = set(), []
    for r in sorted(rows, key=lambda r: -r["val"]):
        key = r["idea"][:60].lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append("  val r %.4f | %.1fM params | %s | %s" % (r["val"], r["n"] / 1e6, recipe(r["tc"]), r["idea"][:110]))
        if len(lines) >= k:
            break
    return title + "\n" + "\n".join(lines)


def trajectory_block(rows):
    cells = sorted({r["cell"] for r in rows})
    blocks = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 99)]
    best = []
    for lo, hi in blocks:
        per = []
        for c in cells:
            v = [r["val"] for r in rows if r["cell"] == c and r["round"] <= hi]
            if v and any(r["cell"] == c and lo <= r["round"] <= hi for r in rows):
                per.append(max(v))
        best.append(np.mean(per) if per else float("nan"))
    early = [r for r in rows if r["round"] <= 5]
    late = [r for r in rows if r["round"] >= 6]

    def mix(rs):
        c = collections.Counter(family(r["idea"]) for r in rs)
        n = max(1, sum(c.values()))
        return ", ".join("%s %d%%" % (k, round(100 * v / n)) for k, v in c.most_common(4))

    def opt_share(rs, name):
        return 100 * np.mean([r["tc"].get("optimizer") == name for r in rs]) if rs else float("nan")

    lines = ["HOW THOSE RUNS PROGRESSED (%d independent runs, rounds of 2 proposals):" % len(cells),
             "  best validation Pearson so far by round block %s: %s" % ("/".join("%d-%d" % (a + 1, min(b, 30) + 1) for a, b in blocks[:-1]) + "/25+",
                                                                      ", ".join("%.3f" % b for b in best)),
             "  mean validation Pearson of proposals: rounds 1-6 %.3f, later rounds %.3f" % (np.mean([r["val"] for r in early]), np.mean([r["val"] for r in late])),
             "  architecture families proposed, rounds 1-6: " + mix(early),
             "  architecture families proposed, later rounds: " + mix(late),
             "  muon was used in %.0f%% of early and %.0f%% of later proposals; median lr early %.0e, later %.0e" % (
                 opt_share(early, "muon"), opt_share(late, "muon"), np.median([float(r["tc"].get("lr", 0)) for r in early]),
                 np.median([float(r["tc"].get("lr", 0)) for r in late]))]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src30", nargs="+", required=True, help="smaller-D AutoResearch instance dirs (e.g. the 30k Sonnet instances)")
    ap.add_argument("--menu10k", required=True, help="menu_retrain dir at the smallest D (ranked by its validation)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_cells(args.src30)
    hdr30 = "RESULTS FROM THE SAME DATASET AT A SMALLER TRAINING-SET SIZE (30,000 sequences; best architectures by validation Pearson across %d independent runs):" % len(args.src30)
    T = top_block(rows, args.k, hdr30) + "\n" + FOOT
    J = top_block(rows, args.k, hdr30) + "\n" + trajectory_block(rows) + "\n" + FOOT
    m10 = []
    for f in glob.glob(args.menu10k.rstrip("/") + "/r*_menu_*_meta.json"):
        m = json.load(open(f))
        if m.get("val_pearson") is not None and m.get("idea"):
            m10.append(dict(cell=0, round=0, val=float(m["val_pearson"]), idea=m["idea"], n=m.get("n_params") or 0, tc=m["train_config"]))
    X = top_block(m10, args.k, "RESULTS FROM THE SAME DATASET AT A SMALLER TRAINING-SET SIZE (10,000 sequences; best architectures by validation Pearson):") + "\n" + FOOT
    for name, txt in (("T", T), ("J", J), ("X", X)):
        (out / f"ctx_{name}.txt").write_text(txt + "\n")
        print(f"=== ctx_{name}.txt ({len(txt)} chars) ===\n{txt}\n")


if __name__ == "__main__":
    main()
