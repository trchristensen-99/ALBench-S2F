"""Where does the designed oligo sit inside an HMEC read, and in which orientation?

This decides the whole counting method. If the oligo starts at a FIXED offset in the read,
every read can be matched against the full design sequence exactly, which assigns the
haplotype siblings that a 30-mer anchor cannot separate (``rs..._alt`` and ``rs..._alt_2``
differ at one other site, and when the two sites are more than K apart no k-mer spans
both -- 29% of the library is unassignable that way). If the offset varies, the anchor has
to supply it per read.

Measures rather than assumes: anchors a sample of reads with a 30-mer whose position in the
design is unambiguous, and reports the implied start offset distribution.

    python3 -I read_structure.py --design a.fa b.fa --fastq rna_rep3_R1.fastq.gz --n 200000
"""

from __future__ import annotations

import argparse
import collections
import sys

import numpy as np

sys.path.insert(0, "scratchpad/nov20/hmec")
from count_oligos import _CODE, _COMP, _open, encode_kmers, read_fasta  # noqa: E402

K = 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", nargs="+", required=True)
    ap.add_argument("--fastq", required=True)
    ap.add_argument("--n", type=int, default=200_000)
    a = ap.parse_args()

    by_seq: dict[str, list[str]] = collections.defaultdict(list)
    for p in a.design:
        for n, s in read_fasta(p):
            by_seq[s.upper()].append(n)
    seqs = sorted(by_seq)
    lens = np.array([len(s) for s in seqs], np.int32)
    width = int(lens.max())
    flat = bytearray()
    for s in seqs:
        flat += s.encode("ascii", "replace").ljust(width, b"N")
    codes = _CODE[np.frombuffer(bytes(flat), np.uint8).reshape(len(seqs), width)]

    # k-mer -> position in the design, kept only where every owner agrees on the position.
    keys_l, pos_l = [], []
    for j in range(width - K + 1):
        q = encode_kmers(codes, j)
        ok = (lens >= j + K) & (q != np.uint64(0xFFFFFFFFFFFFFFFF))
        keys_l.append(q[ok])
        pos_l.append(np.full(int(ok.sum()), j, np.int32))
    keys = np.concatenate(keys_l)
    pos = np.concatenate(pos_l)
    order = np.argsort(keys, kind="stable")
    keys, pos = keys[order], pos[order]
    firstof = np.r_[True, keys[1:] != keys[:-1]]
    gid = np.cumsum(firstof) - 1
    mx = np.zeros(gid[-1] + 1, np.int32)
    np.maximum.at(mx, gid, pos)
    mn = np.full(gid[-1] + 1, np.iinfo(np.int32).max, np.int32)
    np.minimum.at(mn, gid, pos)
    sel = firstof & (mx == mn)[gid]
    keys, pos = keys[sel], pos[sel]
    print("%d design sequences; %d position-unambiguous %d-mers" % (len(seqs), len(keys), K))

    reads: list[str] = []
    with _open(a.fastq) as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                reads.append(line.strip())
                if len(reads) >= a.n:
                    break
    rw = max(len(r) for r in reads)
    print(
        "%d reads sampled, read length %s"
        % (len(reads), collections.Counter(len(r) for r in reads).most_common(3))
    )
    flat = bytearray()
    for r in reads:
        flat += r.encode("ascii", "replace").ljust(rw, b"N")
    rcodes = _CODE[np.frombuffer(bytes(flat), np.uint8).reshape(len(reads), rw)]
    rrc = _COMP[np.minimum(rcodes, 4)][:, ::-1]

    for label, arr in (("forward", rcodes), ("revcomp", rrc)):
        starts = collections.Counter()
        hit = 0
        for o in range(0, rw - K + 1, 4):
            q = encode_kmers(arr, o)
            p = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
            ok = keys[p] == q
            hit += int(ok.sum())
            for s in o - pos[p][ok]:
                starts[int(s)] += 1
        top = starts.most_common(6)
        tot = sum(starts.values())
        print(
            "\n%s: %d probe hits over %d reads; implied oligo start offsets:"
            % (label, hit, len(reads))
        )
        for s, c in top:
            print("   start=%-5d %9d  (%.1f%% of hits)" % (s, c, 100 * c / max(1, tot)))


if __name__ == "__main__":
    main()
