"""Read the Wednesday programme's results off disk: Arm A, Arm B, LOCO, the search.

One reader for all four arms, because they all write the same ``*_meta.json`` shape and a
table assembled by four different ad-hoc scripts is four chances to mismatch a column.

Two conventions it has to respect. The trainer names its own output directory from the
pool basename and that name is not always the reservoir name -- `evoaug_ours_default`
lands in `..._evoaug_ours` -- so directories are resolved by globbing the output root and
reading each member's stamped ``pool`` field, never by rebuilding the name from the cell.
And a bank is reported only when it has the full seed count, because the ensemble gain is
most of the between-reservoir signal and a 3-seed bank is not comparable with a 5-seed one.

    python scratchpad/nov23/read_wed.py --what armA armB loco
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

R = "/grid/wsbs/home_norepl/christen/ALBench-S2F"
ENDPOINTS = ["test_genomic", "test_ood", "test_snv_ref", "test_random_32k"]


def members(d: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(d, "r*_menu_*_meta.json"))):
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def row(ms: list[dict]) -> dict:
    """Per-endpoint mean and spread across seeds, plus the pool each seed actually read."""
    r = {"n_seeds": len(ms)}
    for e in ["val_pearson"] + ENDPOINTS:
        v = [m["report"][e] for m in ms if e in m.get("report", {})]
        if e == "val_pearson":
            v = [m[e] for m in ms if e in m]
        if v:
            r[e] = round(float(np.mean(v)), 4)
            r[e + "_sd"] = round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4)
    pools = {os.path.basename(m.get("pool", "") or "genomic_default") for m in ms}
    r["pool"] = sorted(pools)[0] if len(pools) == 1 else "MIXED:" + ",".join(sorted(pools))
    r["epochs"] = sorted({m.get("best_epoch") for m in ms})
    return r


def table(title: str, rows: dict, key: str = "test_genomic", min_seeds: int = 1) -> None:
    print("\n=== %s" % title)
    hdr = "%-46s %6s" % ("arm", "seeds")
    for e in ["val_pearson"] + ENDPOINTS:
        hdr += "%16s" % e.replace("test_", "")
    print(hdr)
    ok = {k: v for k, v in rows.items() if v["n_seeds"] >= min_seeds}
    for name, r in sorted(ok.items(), key=lambda kv: -kv[1].get(key, -9)):
        line = "%-46s %6d" % (name, r["n_seeds"])
        for e in ["val_pearson"] + ENDPOINTS:
            line += "%16s" % ("%.4f+-%.3f" % (r[e], r[e + "_sd"]) if e in r else "-")
        print(line)
    skipped = len(rows) - len(ok)
    if skipped:
        print("(%d arms below %d seeds, not shown)" % (skipped, min_seeds))


def arm_a() -> None:
    for size, want in (("30k", 5), ("300k", 3)):
        rows = {}
        for d in sorted(glob.glob(os.path.join(R, "outputs/wed41/bench38_ln_*_%s_c20*" % size))):
            ms = members(d)
            if ms:
                rows[os.path.basename(d)] = row(ms)
        table(
            "ARM A %s -- random acquisition = the reservoir's own nested pool" % size,
            rows,
            min_seeds=1,
        )
        short = {k: v["n_seeds"] for k, v in rows.items() if v["n_seeds"] < want}
        if short:
            print("INCOMPLETE (want %d seeds): %s" % (want, short))


def arm_b() -> None:
    rows = {}
    for res in sorted(os.listdir(os.path.join(R, "outputs/acq41"))):
        p = os.path.join(R, "outputs/acq41", res)
        if not os.path.isdir(p) or res in ("logs", "pools"):
            continue
        for arm in sorted(os.listdir(p)):
            for d in glob.glob(os.path.join(p, arm, "bench38_*")):
                ms = members(d)
                if ms:
                    rows["%s/%s" % (res, arm)] = row(ms)
    table("ARM B -- 30k seed set + 30k acquired, 5-member final bank", rows, min_seeds=1)

    # What each rule actually selected. BADGE and BatchBALD can return a full, plausible
    # batch while selecting at random, and the label mean is the cheapest tell.
    print("\n--- selection provenance (label mean of the acquired rows vs the pool) ---")
    print("%-52s %10s %10s %12s" % ("pool", "selected", "pool_mean", "n_informative"))
    for f in sorted(glob.glob(os.path.join(R, "outputs/acq41/pools/*_meta.json"))):
        m = json.load(open(f))
        print(
            "%-52s %10s %10s %12s"
            % (
                os.path.basename(f).replace("_meta.json", ""),
                m.get("label_mean_selected"),
                m.get("label_mean_pool"),
                m.get("n_informative_", "-"),
            )
        )


def loco() -> None:
    """ct44: hold one cell out entirely, 7 folds x 3 modes x 2 seeds."""
    by = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(R, "outputs/celltype/ct44_*"))):
        f = os.path.join(d, "metrics.json")
        if not os.path.exists(f):
            continue
        m = json.load(open(f))
        name = os.path.basename(d)
        mode = name.split("_hold")[0].replace("ct44_", "")
        cell = name.split("_hold")[1].rsplit("_s", 1)[0]
        by[(mode, cell)].append(m)
    if not by:
        print("\n=== ct44 LOCO: no metrics.json yet")
        return
    cells = sorted({c for _, c in by})
    modes = sorted({m for m, _ in by})

    # The held-out cell's own column of the per-cell score block. `designed` carries only
    # the three cells the designed libraries assayed, so four of the seven folds have no
    # designed column at all -- a blank there is a missing measurement, not a zero.
    def finite(xs):
        """Drop non-finite scores and SAY how many were dropped.

        np.mean over a list holding one NaN returns NaN, which prints as a dash-like
        `nan` and reads as "this arm failed" when in fact one seed of two is missing.
        """
        ok = [float(x) for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]
        return ok, len(xs) - len(ok)

    def held_score(m, block, cell):
        v = (m.get(block) or {}).get(cell)
        if isinstance(v, dict):
            return v.get("r") if v.get("zero_shot_valid", True) else None
        return v

    dropped = 0
    for block in ("genomic_test", "designed"):
        print("\n=== ct44 LOCO -- %s, r on the cell that was held out" % block)
        print("%-16s" % "mode" + "".join("%13s" % c for c in cells))
        for mo in modes:
            line = "%-16s" % mo
            for c in cells:
                ok, nbad = finite([held_score(m, block, c) for m in by.get((mo, c), [])])
                dropped += nbad
                line += "%13s" % ("%.4f" % np.mean(ok) if ok else "-")
            print(line)

    # The seen-cell average is the comparison that makes the held-out column readable: a
    # mechanism can only be said to fail on a new cell if it worked on the trained ones.
    print("\n--- same models, mean r over the six cells they DID train on (genomic) ---")
    print("%-16s" % "mode" + "".join("%13s" % c for c in cells))
    for mo in modes:
        line = "%-16s" % mo
        for c in cells:
            vals = []
            for m in by.get((mo, c), []):
                seen, _ = finite(
                    [
                        (v["r"] if isinstance(v, dict) else v)
                        for k, v in (m.get("genomic_test") or {}).items()
                        if k != c
                    ]
                )
                if seen:
                    vals.append(float(np.mean(seen)))
            line += "%13s" % ("%.4f" % np.mean(vals) if vals else "-")
        print(line)
    print("\n%d non-finite or invalid per-cell scores were dropped." % dropped)
    print(
        "%d runs read, %d (mode, holdout) combinations."
        % (sum(len(v) for v in by.values()), len(by))
    )
    print("\nBlank = no valid zero-shot score: `embed` and `onehot` cannot build a vector")
    print("for a cell they never trained, so their held-out row is an untrained artifact.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="+", default=["armA", "armB", "loco"])
    a = ap.parse_args()
    if "armA" in a.what:
        arm_a()
    if "armB" in a.what:
        arm_b()
    if "loco" in a.what:
        loco()


if __name__ == "__main__":
    main()
