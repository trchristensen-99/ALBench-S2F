"""Count HMEC MPRA reads per designed oligo, straight from the fastq.

ENCODE released reads for this experiment but no quantification and no barcode-to-oligo
association map, so the usual MPRA route (barcode -> oligo, then count barcodes) is closed.
It is also unnecessary: the read structure, measured in ``read_structure.py`` rather than
assumed, is that R1 reads the oligo from its 3' end starting at read position 0, so the
designed oligo is exactly ``revcomp(read[:L])`` for the design length L. 71% of anchor hits
imply start offset 49 in the reverse complement of a 150 bp read, which is 150 - 101.

That matters because it makes the match EXACT and therefore unambiguous. A k-mer anchor
cannot do this job here: the library is a haplotype library, so ``rs..._alt`` and
``rs..._alt_2`` are the same allele on two backgrounds and differ at one other site, and
whenever the two sites are more than K apart no k-mer spans both. 29% of the distinct
sequences have no identifying 30-mer for exactly that reason, and they are not a random 29%
-- they are the multi-variant haplotypes. Matching the whole oligo separates all of them.

A sequencing error makes a read MISS rather than be misassigned, which is the safe failure
direction; the miss rate is reported so it can be checked against the ~10% expected for a
101 bp exact match at Q30.

    python3 -I count_oligos.py --design a.fa b.fa --fastq rna_rep1_R1.fastq.gz \
        --out outputs/hmec/counts/rna_rep1.npz
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import time

import numpy as np

_CODE = np.full(256, 255, np.uint8)
for _i, _b in enumerate("ACGT"):
    _CODE[ord(_b)] = _i
    _CODE[ord(_b.lower())] = _i
_COMP = np.array([3, 2, 1, 0, 255], np.uint8)
_BASE = np.uint64(1099511628211)  # FNV-style odd multiplier; any odd 64-bit base works
_BAD = np.uint64(0xFFFFFFFFFFFFFFFF)


def _open(path: str):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gz else open(path)


def read_fasta(path: str):
    name, buf = None, []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line)
    if name is not None:
        yield name, "".join(buf)


def poly_hash(win: np.ndarray) -> np.ndarray:
    """(n, L) base codes -> (n,) uint64 hash; ``_BAD`` wherever the window contains an N."""
    bad = (win == 255).any(1)
    h = np.zeros(len(win), np.uint64)
    with np.errstate(over="ignore"):
        for j in range(win.shape[1]):
            h = h * _BASE + win[:, j].astype(np.uint64) + np.uint64(1)
    h[h == _BAD] = _BAD - np.uint64(1)  # keep the sentinel reserved
    h[bad] = _BAD
    return h


def pack(seqs: list[str], width: int) -> np.ndarray:
    flat = bytearray()
    for s in seqs:
        flat += s.encode("ascii", "replace").ljust(width, b"N")
    return _CODE[np.frombuffer(bytes(flat), np.uint8).reshape(len(seqs), width)]


def build_index(paths: list[str], min_len: int = 90):
    """Distinct design sequences, hashed whole, grouped by length.

    Designs shorter than ``min_len`` are dropped. They are a tiny tail (the library is
    96-101 bp), but they are a trap: a long read carrying a sequencing error fails its own
    101-mer and can then match a short design that happens to equal the end of some other
    oligo, which turns a harmless miss into a wrong assignment. Excluding them keeps the
    failure mode one-directional.
    """
    by_seq: dict[str, list[str]] = collections.defaultdict(list)
    for p in paths:
        for n, s in read_fasta(p):
            by_seq[s.upper()].append(n)
    dropped = sum(1 for s in by_seq if len(s) < min_len)
    if dropped:
        by_seq = {s: v for s, v in by_seq.items() if len(s) >= min_len}
        print("dropped %d design sequences shorter than %d bp" % (dropped, min_len), flush=True)
    seqs = sorted(by_seq)
    names = [by_seq[s] for s in seqs]
    lens = np.array([len(s) for s in seqs], np.int32)
    print(
        "%d design records -> %d distinct sequences; lengths %s"
        % (
            sum(len(v) for v in names),
            len(seqs),
            collections.Counter(lens.tolist()).most_common(6),
        ),
        flush=True,
    )

    tables = {}
    for L in sorted(set(lens.tolist())):
        sel = np.flatnonzero(lens == L)
        h = poly_hash(pack([seqs[i] for i in sel], L))
        if len(np.unique(h)) != len(h):
            raise SystemExit(f"hash collision among the {len(h)} designs of length {L}")
        order = np.argsort(h, kind="stable")
        tables[L] = (h[order], sel[order].astype(np.int32))
    print("   hashed %d length classes, all collision-free" % len(tables), flush=True)
    return tables, seqs, names


def count_fastq(path: str, tables: dict, n_entity: int, *, chunk: int = 400_000, limit: int = 0):
    counts = np.zeros(n_entity, np.int64)
    stat: collections.Counter = collections.Counter()
    t0 = time.time()
    lengths = sorted(tables, reverse=True)

    def flush(block: list[str]) -> None:
        if not block:
            return
        width = max(len(s) for s in block)
        codes = pack(block, width)
        stat["reads"] += len(block)
        done = np.zeros(len(block), bool)
        for L in lengths:
            if width < L:
                continue
            # R1 reads the oligo from its 3' end at read position 0, so the design is the
            # reverse complement of the read's first L bases.
            win = _COMP[np.minimum(codes[:, :L], 4)][:, ::-1]
            h = poly_hash(win)
            keys, owners = tables[L]
            p = np.minimum(np.searchsorted(keys, h), len(keys) - 1)
            ok = (keys[p] == h) & (h != _BAD) & ~done
            if ok.any():
                counts[:] += np.bincount(owners[p[ok]], minlength=n_entity)
                stat["L%d" % L] += int(ok.sum())
                done |= ok
        stat["assigned"] += int(done.sum())

    block: list[str] = []
    with _open(path) as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1:
                continue
            block.append(line.strip())
            if len(block) >= chunk:
                flush(block)
                block = []
                print(
                    "   %11d reads  assigned %.1f%%  (%.0fs)"
                    % (
                        stat["reads"],
                        100 * stat["assigned"] / max(1, stat["reads"]),
                        time.time() - t0,
                    ),
                    flush=True,
                )
                if limit and stat["reads"] >= limit:
                    break
    flush(block)
    return counts, dict(stat), round(time.time() - t0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", nargs="+", required=True)
    ap.add_argument("--fastq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0, help="stop after N reads (smoke test)")
    a = ap.parse_args()

    tables, seqs, names = build_index(a.design, min_len=a.min_len)
    print("counting %s" % a.fastq, flush=True)
    counts, stat, secs = count_fastq(a.fastq, tables, len(seqs), limit=a.limit)

    n = stat.get("reads", 0)
    print(
        "\n%s: %d reads, assigned %d (%.2f%%) in %.0fs"
        % (
            os.path.basename(a.fastq),
            n,
            stat.get("assigned", 0),
            100 * stat.get("assigned", 0) / max(1, n),
            secs,
        ),
        flush=True,
    )
    print("   by design length: %s" % {k: v for k, v in sorted(stat.items()) if k.startswith("L")})
    nz = counts > 0
    print(
        "   %d of %d designs seen; median count %d, total assigned %d"
        % (
            int(nz.sum()),
            len(counts),
            int(np.median(counts[nz])) if nz.any() else 0,
            int(counts.sum()),
        )
    )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        counts=counts,
        sequences=np.array(seqs, dtype=object),
        names=np.array([";".join(v) for v in names], dtype=object),
        fastq=os.path.basename(a.fastq),
        design=np.array([os.path.basename(p) for p in a.design], dtype=object),
        stat=json.dumps(stat),
    )
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
