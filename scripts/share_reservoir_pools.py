"""Publish the labelled reservoir pools and the full evaluation battery to the shared lab folder.

Until now the shared folder held the three headline test sets and no candidate pools at all, so a collaborator running
acquisition experiments had the evaluation half of the study and none of the data half. This copies, with provenance:

  reservoir_pools/   one .npz per reservoir, the canonical draw (generation seed 42), at the largest size that exists.
                     Every smaller size is a PREFIX, so `seqs[:n]` is the study's n-sequence pool for that reservoir --
                     which is the one thing an acquisition experiment must not get wrong, because reshuffling turns a
                     size comparison into a different sample.
  data/eval_sets/    all 19 evaluation sets, not just the three in data/test_sets (which stay where they are).

Each copy is checksummed into MANIFEST.json together with the label mean/sd, GC and the exact source path, because a
pool whose provenance is unknowable is a pool nobody can trust a year from now.

    python scripts/share_reservoir_pools.py [--dry] [--only reservoir_pools|eval_sets]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NEST = REPO / "outputs/grid_pools/k562_nested"
REPS = REPO / "outputs/grid_pools/k562_reps"
BATTERY = REPO / "data/k562/test_sets_ag_s2_chrsplit"
SHARE = Path("/grid/koo/home/shared/albench")

# strategy -> (size to publish, one line for the README)
CORE = {
    "random": (300_000, "uniform random 200-mers; the null every other reservoir has to beat"),
    "genomic": (300_000, "real human CREs from the Gosai library, holdout chromosomes removed"),
    "evoaug_paper2023": (
        300_000,
        "EvoAug-style structural edits (insert/delete/translocate/invert) of real CREs",
    ),
    "motif_ct_enriched": (300_000, "synthetic sequences planted with K562-enriched CIS-BP motifs"),
    "motif_shared_core": (
        300_000,
        "synthetic sequences built from the motif core shared across cell types",
    ),
    "mutagenesis": (300_000, "point mutations of real CREs at the study's default rate"),
}
EXTRA = {
    "zoonomia_orthologs": (
        300_000,
        "real orthologous CRE windows from the 241-mammal Zoonomia alignment "
        "(rebuilt Nov 8 2026; every draw before that was 15.4% exact repeats)",
    ),
    "gc_matched": (300_000, "random sequences matched to the GC distribution of the genomic pool"),
    "motif_syntax_core": (
        300_000,
        "shared-core motifs placed with learned spacing/orientation syntax",
    ),
    "hg38_random": (
        100_000,
        "random hg38 windows (not CRE-selected), holdouts removed by sequence",
    ),
    "mutagenesis_mr10": (100_000, "mutagenesis at a 10% per-base rate"),
    "random_gc_sweep": (100_000, "random sequences swept across GC content"),
}
# `zoonomia` (the old conservation-RATE arm) is not published at all -- the study dropped it in favour of
# the real alignment sequences, which are what `zoonomia_orthologs` now holds.
PENDING: dict[str, tuple[int, str]] = {}
SKIP_EVAL = ("snv_mono_chr7_13_intermediate",)  # a 70 MB intermediate, not an evaluation set


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def describe(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    # the SNV set is the one that stores PAIRS rather than a flat list, so it names its columns differently
    skey = next(k for k in ("sequences", "ref_sequences") if k in z.files)
    key = next(k for k in ("oracle_labels", "oracle_mean", "ref_mean") if k in z.files)
    seqs = z[skey]
    y = z[key].astype(np.float32)
    sample = [str(s) for s in seqs[:20000]]
    return {
        "n": int(len(seqs)),
        "n_distinct": int(len(set(str(s) for s in seqs))),
        "seq_len": int(len(sample[0])),
        "seq_key": skey,
        "label_key": key,
        "label_mean": round(float(y.mean()), 4),
        "label_sd": round(float(y.std()), 4),
        "gc": round(float(np.mean([(s.count("G") + s.count("C")) / len(s) for s in sample])), 4),
        "oracle_id": str(z["oracle_id"]) if "oracle_id" in z.files else "unknown",
    }


def publish(src: Path, dest: Path, dry: bool) -> dict:
    info = describe(src)
    info["source"] = str(src)
    if not dry:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        shutil.copyfile(src, tmp)
        os.chmod(tmp, 0o644)
        tmp.replace(dest)
        info["sha256_16"] = sha256(dest)
    dup = 100 * (1 - info["n_distinct"] / info["n"])
    print(
        f"  {dest.name:42s} n={info['n']:>7,}  gc={info['gc']:.3f}  "
        f"label {info['label_mean']:+.3f}+-{info['label_sd']:.3f}"
        + (f"  DUP {dup:.1f}%" if dup > 0.5 else "")
        + ("  [DRY]" if dry else "")
    )
    return info


def do_pools(dry: bool) -> dict:
    out = SHARE / "reservoir_pools"
    man = {}
    for group, table in (("core", CORE), ("additional", EXTRA), ("pending", PENDING)):
        for strat, (size, doc) in table.items():
            src = NEST / f"{strat}_d{size}.npz"
            if group == "pending":
                print(f"  {strat}: held back until the deduplicated rebuild lands")
                continue
            if not src.exists():
                print(f"  {strat}: no canonical pool at {size:,}")
                continue
            info = publish(src, out / src.name, dry)
            info.update(group=group, doc=doc, strategy=strat)
            # which study sizes are available as prefixes of this file
            info["prefix_sizes"] = [s for s in (10_000, 30_000, 100_000, 300_000) if s <= info["n"]]
            man[src.name] = info
    return man


def do_replicates(dry: bool) -> dict:
    """The three independent draws per reservoir, which is what error bars on a corpus comparison need.

    One draw tells you how a reservoir scored; three tell you whether the difference between two reservoirs
    is bigger than the difference between two draws of the same one. Published under `replicates/` rather
    than beside the canonical pools so nobody mixes a replicate into a canonical result by accident.
    """
    out = SHARE / "reservoir_pools/replicates"
    man = {}
    for strat, (size, _doc) in {**CORE, **EXTRA}.items():
        for rep in (1, 2, 3):
            src = REPS / f"{strat}_r{rep}_d{size}.npz"
            if not src.exists():  # fall back to the largest replicate size that was built
                smaller = sorted(
                    REPS.glob(f"{strat}_r{rep}_d*.npz"), key=lambda p: int(p.stem.split("_d")[1])
                )
                if not smaller:
                    continue
                src = smaller[-1]
            info = publish(src, out / src.name, dry)
            info.update(strategy=strat, replicate=rep)
            info["prefix_sizes"] = [s for s in (10_000, 30_000, 100_000, 300_000) if s <= info["n"]]
            man[src.name] = info
    return man


def do_eval(dry: bool) -> dict:
    out = SHARE / "data/eval_sets"
    man = {}
    for src in sorted(BATTERY.glob("*.npz")):
        stem = src.stem
        if stem.endswith((".premix", ".bak")) or "legacy" in stem or stem in SKIP_EVAL:
            continue
        if src.stat().st_size < 50_000:  # an empty or placeholder set helps nobody
            print(f"  {stem}: {src.stat().st_size} bytes, skipped")
            continue
        info = publish(src, out / src.name, dry)
        info["set"] = stem.replace("_oracle", "")
        man[src.name] = info
    return man


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", choices=("reservoir_pools", "replicates", "eval_sets"))
    args = ap.parse_args()
    out = {"written_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if args.only in (None, "reservoir_pools"):
        print("reservoir pools ->", SHARE / "reservoir_pools")
        out["reservoir_pools"] = do_pools(args.dry)
    if args.only in (None, "replicates"):
        print("replicate draws ->", SHARE / "reservoir_pools/replicates")
        out["replicates"] = do_replicates(args.dry)
    if args.only in (None, "eval_sets"):
        print("evaluation sets ->", SHARE / "data/eval_sets")
        out["eval_sets"] = do_eval(args.dry)
    if not args.dry:
        for sub, key in (
            ("reservoir_pools", "reservoir_pools"),
            ("reservoir_pools/replicates", "replicates"),
            ("data/eval_sets", "eval_sets"),
        ):
            if key in out:
                p = SHARE / sub / "MANIFEST.json"
                p.write_text(
                    json.dumps({"written_utc": out["written_utc"], key: out[key]}, indent=1)
                )
                os.chmod(p, 0o644)
                print("wrote", p)


if __name__ == "__main__":
    main()
