"""Publish the pools that the Nov-8 share is missing, into the same shared folder.

The existing share publishes ONE file per reservoir at the largest size and relies on the
pools being nested, so `sequences[:30000]` is exactly the 30k pool. That stays; this adds
what a mixed-reservoir scaling study needs and the Nov-8 cut did not have: the two 1M-scale
reservoirs, the mixture pools, and the strategies added since.

Nesting is the one property a reader cannot check cheaply and the whole slicing convention
rests on it, so it is VERIFIED here per strategy rather than assumed -- a mixture family
whose sizes turn out to be independent draws is published at every size instead, with the
README saying so.

Nothing is overwritten: a destination that already exists is left alone and reported.

    python3 -I stage_shared_pools.py --dry-run
    python3 -I stage_shared_pools.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import time

import numpy as np

SRC = "/grid/wsbs/home_norepl/christen/ALBench-S2F/outputs/grid_pools"
DST = "/grid/koo/home/shared/albench/reservoir_pools"

# Canonical single-reservoir pools the Nov-8 cut does not carry.
WANTED = [
    "motif_shared_core",
    "motif_ct_enriched",
    "dinuc_shuffle",
    "evoaug_ours_default",
    "evo2",
    "d3_steps64",
    "d3_steps256",
    "accessible_k562_only",
    "accessible_hepg2_only",
    "accessible_tiles_k562",
]


def sizes(sub: str, strategy: str) -> list[tuple[int, str]]:
    out = []
    for p in glob.glob(os.path.join(SRC, sub, f"{strategy}_d*.npz")):
        tail = p.rsplit("_d", 1)[1][:-4]
        if tail.isdigit():
            out.append((int(tail), p))
    return sorted(out)


def check_nested(chain: list[tuple[int, str]]) -> tuple[bool, str]:
    """Is each size a prefix of the next? Returns (ok, note)."""
    prev = None
    for _d, p in chain:
        s = np.load(p, allow_pickle=True)["sequences"]
        if prev is not None and not np.array_equal(s[: len(prev)], prev):
            return False, "sizes are independent draws, not a prefix chain"
        prev = s
    return True, "prefix chain verified across %d sizes" % len(chain)


def describe(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    seq = z["sequences"]
    y = z["oracle_labels"]
    gc = np.mean([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seq[:20000]])
    return {
        "n": int(len(seq)),
        "n_distinct": int(len(set(map(str, seq)))),
        "gc": round(float(gc), 3),
        "label_mean": round(float(np.mean(y)), 3),
        "label_sd": round(float(np.std(y)), 3),
        "strategy": str(z["strategy"]) if "strategy" in z.files else "",
        "params": str(z["params"]) if "params" in z.files else "",
        "seed": int(z["seed"]) if "seed" in z.files else None,
        "oracle_id": str(z["oracle_id"]) if "oracle_id" in z.files else "",
    }


def sha256(path: str, limit: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            if fh.tell() > limit:
                break
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    plan: list[tuple[str, str, dict]] = []

    print("=== single-reservoir pools")
    for s in WANTED:
        chain = sizes("k562_nested", s)
        if not chain:
            print("   %-24s MISSING" % s)
            continue
        ok, note = check_nested(chain)
        d, src = chain[-1]
        dst = os.path.join(DST, os.path.basename(src))
        meta = describe(src)
        meta["nested"] = ok
        meta["nesting_note"] = note
        meta["all_sizes"] = [x[0] for x in chain]
        state = "exists" if os.path.exists(dst) else "ADD"
        print(
            "   %-24s d=%-8d %-6s %s | %s"
            % (s, d, state, note, "distinct %d/%d" % (meta["n_distinct"], meta["n"]))
        )
        if state == "ADD":
            plan.append((src, dst, meta))

    print("\n=== mixture pools")
    fams = sorted(
        {
            os.path.basename(p).rsplit("_d", 1)[0]
            for p in glob.glob(os.path.join(SRC, "k562_mixtures", "*_d*.npz"))
        }
    )
    os.makedirs(os.path.join(DST, "mixtures"), exist_ok=True) if not a.dry_run else None
    for f in fams:
        chain = sizes("k562_mixtures", f)
        ok, note = check_nested(chain)
        # A family that is NOT nested has to go out at every size or the reader cannot
        # reproduce the smaller points at all.
        pub = [chain[-1]] if ok else chain
        for d, src in pub:
            dst = os.path.join(DST, "mixtures", os.path.basename(src))
            if os.path.exists(dst):
                continue
            meta = describe(src)
            meta["nested"] = ok
            meta["nesting_note"] = note
            meta["all_sizes"] = [x[0] for x in chain]
            plan.append((src, dst, meta))
        print("   %-22s sizes=%-28s %s" % (f, [x[0] for x in chain], note))

    total = sum(os.path.getsize(s) for s, _, _ in plan)
    print("\n%d files to publish, %.1f GB" % (len(plan), total / 1e9))
    if a.dry_run:
        return

    manifest = []
    for i, (src, dst, meta) in enumerate(plan, 1):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        if os.path.getsize(dst) != os.path.getsize(src):
            raise SystemExit("short copy: %s" % dst)
        np.load(dst, allow_pickle=True)["sequences"]  # it must open where it landed
        os.chmod(dst, 0o644)
        meta.update(
            {
                "file": os.path.relpath(dst, DST),
                "bytes": os.path.getsize(dst),
                "sha256_head": sha256(dst),
                "source": src,
                "published": time.strftime("%Y-%m-%d"),
            }
        )
        manifest.append(meta)
        print("   %2d/%d %s" % (i, len(plan), meta["file"]), flush=True)

    mpath = os.path.join(DST, "MANIFEST_nov21.json")
    with open(mpath, "w") as fh:
        json.dump({"published": time.strftime("%Y-%m-%d"), "files": manifest}, fh, indent=1)
    os.chmod(mpath, 0o644)
    print("\nwrote %s (%d entries)" % (mpath, len(manifest)))


if __name__ == "__main__":
    main()
