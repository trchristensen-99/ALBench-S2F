#!/usr/bin/env python
"""Two unlabelled reservoirs drawn from the reference genome rather than from the Gosai library.

`accessible`: 200-bp windows tiled across merged ENCODE DNase peaks of one cell type. The Gosai library holds only ~11k K562-accessible windows, but the peaks themselves cover 14.5 Mb in K562 and 31.1 Mb in
HepG2, so tiling them reaches ~72k and ~155k windows. This is the `encode_accessibility` idea, descoped on 2026-09-09 for not scaling past ~100k, which is still the right ceiling to expect.

`hg38_random`: 200-bp windows at uniformly random positions in the reference, the control arm. It says what a student learns from genomic sequence chosen with no regard to assay, variant or accessibility, so
the accessible and library arms can be read against something that has neither property.

Both exclude the test and validation chromosomes (chr7, chr13 test; chr19, chr21, chrX validation) and any window that is not a clean 200-mer of ACGT, which drops assembly gaps. Labels come afterwards from
`scripts/label_pool.py`, as for every other reservoir.

    python scripts/build_genome_source_pools.py --kind accessible --cell K562 --n 100000 --out outputs/grid_pools/k562_unlabelled/accessible_tiles_k562_d100000.npz
    python scripts/build_genome_source_pools.py --kind hg38_random --n 100000 --out outputs/grid_pools/k562_unlabelled/hg38_random_d100000.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from albench.core.paths import resolve  # noqa: E402

HOLDOUT = {"chr7", "chr13", "chr19", "chr21", "chrX"}
LEN = 200
STEP = 100  # overlapping tiles: a peak of median ~280 bp yields 1-2 windows


class Fasta:
    """Random access through the .fai index: pysam is not in the shared venv and must not be installed while jobs run."""

    def __init__(self, path: str) -> None:
        if not os.path.exists(f"{path}.fai"):
            raise SystemExit(f"need a samtools index beside the reference: {path}.fai")
        self.fh = open(path, "rb")
        self.idx = {}
        for line in open(f"{path}.fai"):
            name, ln, off, bases, width = line.split("\t")[:5]
            self.idx[name] = (int(ln), int(off), int(bases), int(width))

    def lengths(self) -> dict:
        return {
            c: v[0]
            for c, v in self.idx.items()
            if c.startswith("chr")
            and "_" not in c
            and c not in ("chrM", "chrY")
            and c not in HOLDOUT
        }

    def fetch(self, chrom: str, start: int, n: int) -> str:
        ln, off, bases, width = self.idx[chrom]
        if start < 0 or start + n > ln:
            return ""
        self.fh.seek(off + start // bases * width + start % bases)
        raw = self.fh.read(n + n // bases + width)
        return raw.replace(b"\n", b"").replace(b"\r", b"")[:n].decode().upper()


def tiles_from_peaks(which: str) -> list:
    """`which` is a cell (its merged peaks) or one of the precomputed partitions, which are the interesting arms:
    k562_only and shared_open_both are the K562 regulatory sequence, hepg2_only is open somewhere but not in the
    assayed cell type, so it separates "open chromatin is informative" from "open in THIS cell type is informative"."""
    root = resolve("encode_peaks")
    bed = (
        f"{root}/{which}_peaks_merged.bed" if which in ("K562", "HepG2") else f"{root}/{which}.bed"
    )
    out = []
    for line in open(bed):
        if line.startswith(("#", "track")):
            continue
        c, s, e = line.split()[:3]
        if c in HOLDOUT or "_" in c:
            continue
        s, e = int(s), int(e)
        if e - s < LEN:  # centre a single window on a peak narrower than the window
            out.append((c, max(0, (s + e) // 2 - LEN // 2)))
            continue
        out += [(c, p) for p in range(s, e - LEN + 1, STEP)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["accessible", "hg38_random"])
    ap.add_argument(
        "--cell",
        default="K562",
        choices=["K562", "HepG2", "k562_only", "hepg2_only", "shared_open_both"],
        help="a cell's merged peaks, or one of the precomputed partitions",
    )
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fa = Fasta(resolve("hg38"))
    rng = np.random.default_rng(a.seed)
    seqs, seen = [], set()
    if a.kind == "accessible":
        cand = tiles_from_peaks(a.cell)
        print(f"{a.cell}: {len(cand)} candidate windows from the peaks")
        rng.shuffle(cand)
        if len(cand) < a.n:
            print(f"NOTE only {len(cand)} windows exist; writing all of them instead of {a.n}")
    else:
        lens = fa.lengths()
        names = list(lens)
        w = np.array([lens[c] for c in names], float)
        w /= w.sum()  # sample a chromosome in proportion to its length
        draws = 3 * a.n  # over-draw: assembly gaps and holdout filtering remove some
        cand = [
            (names[i], int(rng.integers(0, lens[names[i]] - LEN)))
            for i in rng.choice(len(names), size=draws, p=w)
        ]
    for c, s in cand:
        if len(seqs) >= a.n:
            break
        seq = fa.fetch(c, s, LEN)
        if len(seq) == LEN and set(seq) <= set("ACGT") and seq not in seen:
            seen.add(seq)
            seqs.append(seq)
    print(
        f"wrote {len(seqs)} sequences ({a.kind}{', ' + a.cell if a.kind == 'accessible' else ''})"
    )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        sequences=np.array(seqs),
        strategy=f"accessible_tiles_{a.cell.lower()}" if a.kind == "accessible" else "hg38_random",
        seed=a.seed,
        window=LEN,
        step=STEP,
    )


if __name__ == "__main__":
    main()
