"""Read the acquisition arms (al52): one-shot against iterative, and what each arm actually bought.

Two tables. The first is the thing the library decision turns on -- every arm ends at the same number of
labelled sequences, so the only question is whether the rounds earned their cost. The second is the part
that is interesting whatever the first says: the candidate pool is an equal mix of seven reservoirs, so each
arm's purchases are a vote on corpus composition from a selector that was never told what rp47 found.

    python scripts/analysis/read_acq52.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scratchpad"))
from ens_util import enet_all_pos  # noqa: E402

SETS = ("genomic", "ood", "random_32k")
DONE = REPO / "outputs/menu_retrain_v0/acq52_done"
MR = REPO / "outputs/menu_retrain_v0"
ORDER = ("os_random", "os_epi", "os_badge", "it3_epi", "it3_badge", "os_oracle")


def r(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def read_run(d: Path):
    V, T = [], {s: [] for s in SETS}
    for m in sorted(glob.glob(str(d / "*_meta.json"))):
        z = np.load(m.replace("_meta.json", ".npz"), allow_pickle=True)
        if "test_pred_genomic" not in z.files:
            continue
        V.append(z["val_pred"])
        for s in SETS:
            T[s].append(z["test_pred_" + s])
    if not V:
        return None
    lab = np.load(d / "labels.npz", allow_pickle=True)
    yv = lab["val_labels"]
    Y = {s: lab["oracle_" + s] for s in SETS}
    V = np.array(V).T
    T = {s: np.array(v).T for s, v in T.items()}
    single = {s: float(np.mean([r(T[s][:, i], Y[s]) for i in range(V.shape[1])])) for s in SETS}
    stack, _, _ = enet_all_pos(V, yv, T, np.arange(len(yv)))
    return single, {s: r(stack[s], Y[s]) for s in SETS}, V.shape[1]


def main() -> None:
    # Keyed by DIRECTORY, not by tag. al52, al55 and al56 reuse arm tags ("os_epi_s1") at
    # different endpoints, so a tag-keyed dict silently kept one arm per tag and printed, say,
    # al56's 60k numbers under al52's 105k header.
    traces = {}
    for p in sorted(MR.glob("*/acq_trace.json")):
        t = json.loads(p.read_text())
        traces[p.parent] = (t, p.parent)
    if not traces:
        print("no finished arms yet")
        return

    print("=== every acquisition arm, grouped by the endpoint it actually trains at ===")
    print(
        f"{'arm':14s} {'rep':>3s} {'n':>2s} {'genomic single/stack':>26s} "
        f"{'designed single/stack':>26s} {'random single/stack':>26s}"
    )
    rows: dict[str, list] = {}
    for _, (t, d) in sorted(traces.items(), key=lambda kv: str(kv[0])):
        if t["schedule"] == [0]:
            continue
        got = read_run(d)
        if not got:
            continue
        single, stack, n = got
        tag = t["tag"]
        # What distinguishes two same-named arms is the budget they acquire: al56 buys 30k on a
        # 30k pilot (60k endpoint), al52/al54 buy 75k on 30k (105k), al55 buys 100k on 100k
        # (200k), al53 buys 300k on 300k (600k). The trace does not record the pilot size, so
        # this labels the acquired budget and does not silently assert an endpoint.
        # not every family suffixes the replicate: a tag without `_s` is a single-replicate arm,
        # and splitting it unconditionally took the whole readout down rather than that one row
        base, sep, rep = tag.rpartition("_s")
        if not sep:
            base, rep = tag, "1"
        arm = f"+{sum(t['schedule']) // 1000}k {base}"
        rows.setdefault(arm, []).append((single, stack))
        cols = " ".join(f"{single[s]:.4f}/{stack[s]:.4f}" for s in SETS)
        print(f"{arm:20s} {rep:>3s} {n:>2d}   {cols}")

    print()
    print(f"{'arm':20s} {'reps':>4s} " + " ".join(f"{s + ' stack':>16s}" for s in SETS))
    order = [a for a in rows if a.split(" ", 1)[1] in ORDER] + [
        a for a in rows if a.split(" ", 1)[1] not in ORDER
    ]
    for arm in sorted(order, key=lambda a: (int(a.split("k ", 1)[0].lstrip("+")), a)):
        rs = rows[arm]
        cols = []
        for s in SETS:
            v = np.array([x[1][s] for x in rs])
            cols.append(
                f"{v.mean():.4f}+-{v.std():.4f}" if len(v) > 1 else f"{v.mean():.4f}        "
            )
        print(f"{arm:20s} {len(rs):>4d} " + " ".join(f"{c:>16s}" for c in cols))

    print()
    print("=== what each arm bought, by reservoir (share of its 75,000) ===")
    srcs: list[str] = []
    for t, _ in traces.values():
        for rd in t["rounds"]:
            srcs += list(rd.get("composition", {}))
    srcs = sorted(set(srcs))
    print(
        f"{'arm':14s} {'rep':>3s} {'label mean':>10s} " + " ".join(f"{s[:11]:>12s}" for s in srcs)
    )
    for tag, (t, _d) in sorted(traces.items()):
        if t["schedule"] == [0]:
            continue
        comp: dict[str, int] = {}
        tot = 0
        wsum = 0.0
        for rd in t["rounds"]:
            for k, v in rd.get("composition", {}).items():
                comp[k] = comp.get(k, 0) + v
            if rd.get("n_add"):
                tot += rd["n_add"]
                wsum += rd["n_add"] * rd["sel_label_mean"]
        n = max(1, sum(comp.values()))
        arm, rep = tag.rsplit("_s", 1)
        print(
            f"{arm:14s} {rep:>3s} {wsum / max(1, tot):>10.3f} "
            + " ".join(f"{100 * comp.get(s, 0) / n:>11.1f}%" for s in srcs)
        )
    print("\n(an equal draw would be 14.3% each; the candidate pool is 1/7 of each reservoir)")

    print("\n=== the iterative arms, round by round ===")
    for tag, (t, _d) in sorted(traces.items()):
        if len(t["schedule"]) < 2:
            continue
        print(f"{tag}:")
        for rd in t["rounds"]:
            if rd.get("final"):
                print(f"  final train at {rd['trained_at']:,}")
                continue
            top = sorted(rd.get("composition", {}).items(), key=lambda kv: -kv[1])[:3]
            print(
                f"  round {rd['round']}: model at {rd['trained_at']:>7,} -> picked {rd['n_add']:,} "
                f"| label {rd['sel_label_mean']:+.3f}+-{rd['sel_label_sd']:.3f} | "
                + ", ".join(f"{k} {100 * v / rd['n_add']:.0f}%" for k, v in top)
            )


if __name__ == "__main__":
    main()
