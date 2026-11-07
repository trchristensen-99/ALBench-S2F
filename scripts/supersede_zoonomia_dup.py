"""Swap the deduplicated zoonomia pools in, and move everything the duplicate pools produced aside.

Run after the `zoo7_*` cells finish. For each generation seed it slices the new 300k draw into nested prefixes
(10k / 30k / 100k / 300k), writes them where the study already looks -- `k562_nested` for seed 42, `k562_reps` for
43/44/45 -- and moves the superseded file to `outputs/grid_pools/superseded_zoonomia_dup/`.

Then it moves aside every result directory whose cells read a zoonomia pool. Nothing is deleted: the old numbers are
what tells us what 15.4% duplication cost, and moving rather than deleting is also what makes the dispatcher re-run
those cells against the new pools (it decides done-ness from COMPLETE files inside the output directory).

    python scripts/supersede_zoonomia_dup.py [--dry]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GP = REPO / "outputs/grid_pools"
RAW = GP / "k562_zoo_dedupe_raw300k"
STRAT = "zoonomia_orthologs"
SIZES = (10_000, 30_000, 100_000, 300_000)
# generation seed -> (directory, filename template). 42 is the canonical pool, 43/44/45 the replicates.
DEST = {
    42: (GP / "k562_nested", f"{STRAT}_d{{d}}.npz"),
    43: (GP / "k562_reps", f"{STRAT}_r1_d{{d}}.npz"),
    44: (GP / "k562_reps", f"{STRAT}_r2_d{{d}}.npz"),
    45: (GP / "k562_reps", f"{STRAT}_r3_d{{d}}.npz"),
}
OLD_POOLS = GP / "superseded_zoonomia_dup"
OLD_RESULTS = REPO / "outputs/superseded_zoonomia_dup"


def slice_pools(dry: bool) -> list[Path]:
    OLD_POOLS.mkdir(parents=True, exist_ok=True) if not dry else None
    written = []
    for gseed, (outdir, tmpl) in DEST.items():
        src = sorted(glob.glob(str(RAW / f"{STRAT}__g{gseed}" / "*.npz")))
        if not src or not (RAW / f"{STRAT}__g{gseed}" / "COMPLETE").exists():
            print(f"  g{gseed}: not built yet, skipped")
            continue
        z = np.load(src[0], allow_pickle=True)
        seqs, y, oid = (
            list(z["sequences"]),
            z["oracle_labels"].astype(np.float32),
            str(z["oracle_id"]),
        )
        n_uniq = len(set(seqs))
        print(
            f"  g{gseed}: {len(seqs):,} rows, {n_uniq:,} distinct ({100 * (1 - n_uniq / len(seqs)):.1f}% dup)"
        )
        if n_uniq < len(seqs):
            print("    still duplicated -- refusing to publish; the sampler fix did not take")
            continue
        for d in SIZES:
            if d > len(seqs):
                continue
            dest = outdir / tmpl.format(d=d)
            if dest.exists() and not dry:
                dest.replace(OLD_POOLS / dest.name)
            if not dry:
                tmp = dest.with_suffix(".tmpnew.npz")
                np.savez(
                    tmp,
                    sequences=np.asarray(seqs[:d], dtype=object),
                    oracle_labels=y[:d],
                    oracle_id=oid,
                    strategy=STRAT,
                    seed=np.int64(gseed),
                )
                tmp.replace(dest)
            written.append(dest)
            print(f"    {'would write' if dry else 'wrote'} {dest.relative_to(REPO)}")
    return written


def stale_result_dirs() -> list[str]:
    """Output directories of every registered cell that reads a zoonomia pool."""
    reg = json.load(open(REPO / "scratchpad/dispatch_registry.json"))
    dirs = set()
    for c in reg:
        blob = json.dumps(c.get("needs", [])) + c.get("cmd", "")
        if f"{STRAT}_" in blob and "grid_pools" in blob:
            dirs.add(c["out_dir"])
    return sorted(d for d in dirs if (REPO / d).is_dir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    print("pools:")
    written = slice_pools(args.dry)
    if not written:
        print("nothing written; leaving results alone")
        return
    print("results trained on the duplicate pools:")
    if not args.dry:
        OLD_RESULTS.mkdir(parents=True, exist_ok=True)
    for d in stale_result_dirs():
        # mirror the whole relative path, not just the leaf: several of these directories are called
        # "rep1", and flattening them would collide and silently drop all but the first
        src = REPO / d
        dest = OLD_RESULTS / Path(d).relative_to("outputs")
        n = len(list(src.glob("COMPLETE*")))
        print(
            f"    {'would move' if args.dry else 'moved'} {d} ({n} completed cells) -> {dest.relative_to(REPO)}"
        )
        if not args.dry:
            if dest.exists():
                print("      destination exists, left in place")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dest)


if __name__ == "__main__":
    main()
