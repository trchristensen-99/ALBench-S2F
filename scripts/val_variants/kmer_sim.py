"""Exact canonical k-mer inverted index for "is this sequence a near-copy of anything in a reference collection?".

Why exact and not MinHash: a 200 bp sequence has only ~185 k-mers, and the question is about a handful of shared k-mers
(a 10%-mutated copy still shares ~19% of its 16-mers). A MinHash sketch of ~128 hashes cannot resolve that against ~1.2M
references; the inverted index can, exactly, and fits in memory (~230M postings).

Similarity of a query q to a reference r is   shared(q, r) / n_kmers(q)   where shared counts DISTINCT canonical
(strand-collapsed, so reverse-complement copies and inversions are caught) k-mers present in both. The score of q is its
maximum over all references, and we also return WHICH reference attains it.

Choice of k is made by measurement (build_val_variants.py calibrate): with ~1.2M references and k=12 a typical k-mer already
occurs in ~28 references by chance (4^12 = 16.8M possible, 230M instances), so unrelated sequences share ~7 12-mers with their
best match and copies beyond ~15% divergence become indistinguishable from chance. At k=16 (4.3G possible) the chance floor is
~0 and the same copies stay detectable.

Repeat masking: a k-mer present in more than `cap` reference sequences belongs to a repeat family (Alu, L1, simple repeats).
Two members of a family are related by biology, not by derivation, and counting those k-mers would flag every
Alu-containing sequence. They are ignored in the numerator (the denominator stays the query's full k-mer count, so a masked
sequence scores LOW, never high). Exact duplicates are caught separately by hashing.
"""

from __future__ import annotations

import numpy as np

SENT = np.uint64(0xFFFFFFFFFFFFFFFF)
_LUT = np.full(256, 4, dtype=np.uint8)
for _i, _c in enumerate(b"ACGT"):
    _LUT[_c] = _i
    _LUT[_c + 32] = _i  # lowercase


def encode(seqs, length: int) -> np.ndarray:
    """Strings -> (N, length) uint8 codes (A,C,G,T = 0..3; anything else, including padding, = 4). Truncates/pads to `length`."""
    seqs = [str(s) for s in seqs]
    if all(len(s) == length for s in seqs):
        buf = np.frombuffer("".join(seqs).encode("ascii", "replace"), dtype=np.uint8).reshape(
            len(seqs), length
        )
    else:
        buf = np.frombuffer(
            "".join(s[:length].ljust(length, "N") for s in seqs).encode("ascii", "replace"),
            dtype=np.uint8,
        ).reshape(len(seqs), length)
    return _LUT[buf]


def canonical_kmers(codes: np.ndarray, k: int) -> np.ndarray:
    """(N, L) codes -> (N, L-k+1) uint64 canonical k-mers (k <= 32); invalid (contains N) or repeated-within-row = SENT, rows sorted."""
    n, L = codes.shape
    W = L - k + 1
    bad = codes > 3
    c0 = np.where(bad, 0, codes).astype(np.uint64)
    cs = np.concatenate(
        [np.zeros((n, 1), np.int32), np.cumsum(bad, axis=1, dtype=np.int32)], axis=1
    )
    win_bad = (cs[:, k : k + W] - cs[:, :W]) > 0
    fwd = np.zeros((n, W), np.uint64)
    rc = np.zeros((n, W), np.uint64)
    comp = np.uint64(3) - c0
    two = np.uint64(2)
    for j in range(k):
        fwd = (fwd << two) | c0[:, j : j + W]
        rc |= comp[:, j : j + W] << np.uint64(2 * j)
    can = np.minimum(fwd, rc)
    can[win_bad] = SENT
    can.sort(axis=1)
    dup = np.zeros_like(can, dtype=bool)
    dup[:, 1:] = (can[:, 1:] == can[:, :-1]) & (can[:, 1:] != SENT)
    can[dup] = SENT
    can.sort(axis=1)
    return can


class KmerIndex:
    """Inverted index over a reference collection. `add` rows (each with a source bitmask), `finalize`, then `query`."""

    def __init__(self, k: int = 16, length: int = 200) -> None:
        self.k, self.length = k, length
        self._keys: list[np.ndarray] = []
        self._ids: list[np.ndarray] = []
        self._src: list[np.ndarray] = []
        self.n_ref = 0

    def add(self, seqs, masks, chunk: int = 50_000) -> None:
        """Add reference rows. `masks` is a uint64 per row: which named sources contain this exact sequence."""
        seqs = list(seqs)
        masks = np.asarray(masks, dtype=np.uint64)
        assert len(seqs) == len(masks)
        for s in range(0, len(seqs), chunk):
            codes = encode(seqs[s : s + chunk], self.length)
            can = canonical_kmers(codes, self.k)
            m = can != SENT
            self._keys.append(can[m])
            self._ids.append(
                np.repeat(np.arange(self.n_ref, self.n_ref + len(codes), dtype=np.uint32), m.sum(1))
            )
            self._src.append(masks[s : s + chunk])
            self.n_ref += len(codes)

    def finalize(self) -> None:
        keys = np.concatenate(self._keys)
        ids = np.concatenate(self._ids)
        self._keys, self._ids = [], []
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        self.ids = ids[order]
        del ids, order
        starts = np.concatenate([[0], np.flatnonzero(keys[1:] != keys[:-1]) + 1]).astype(np.int64)
        self.ukeys = keys[starts]
        self.counts = np.diff(np.concatenate([starts, [len(keys)]])).astype(np.int64)
        self.starts = starts
        del keys
        self.src = np.concatenate(self._src)
        self._src = []

    def save(self, d) -> None:
        from pathlib import Path

        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for name in ("ids", "ukeys", "counts", "starts", "src"):
            np.save(d / f"{name}.npy", getattr(self, name))
        (d / "k.txt").write_text(f"{self.k} {self.length}\n")

    @classmethod
    def load(cls, d) -> "KmerIndex":
        from pathlib import Path

        d = Path(d)
        k, length = (int(x) for x in (d / "k.txt").read_text().split())
        o = cls(k, length)
        for name in ("ids", "ukeys", "counts", "starts", "src"):
            setattr(o, name, np.load(d / f"{name}.npy"))
        o.n_ref = len(o.src)
        return o

    def query(self, seqs, cap: int = 10**9, max_postings: int = 30_000_000):
        """Return (shared[int32], n_kmers[int32], best_ref[int64], n_masked[int32]) for every query sequence.

        `shared` = max over references of #distinct canonical k-mers in common, ignoring k-mers present in more than
        `cap` reference sequences (repeat families). `n_kmers` is the query's distinct valid k-mer count.
        """
        seqs = list(seqs)
        nq = len(seqs)
        shared = np.zeros(nq, np.int32)
        best = np.full(nq, -1, np.int64)
        nk = np.zeros(nq, np.int32)
        nmask = np.zeros(nq, np.int32)
        for s0 in range(0, nq, 20_000):
            codes = encode(seqs[s0 : s0 + 20_000], self.length)
            can = canonical_kmers(codes, self.k)
            valid = can != SENT
            pos = np.searchsorted(self.ukeys, can)
            pos = np.minimum(pos, len(self.ukeys) - 1)
            hit = valid & (self.ukeys[pos] == can)
            cnt = np.where(hit, self.counts[pos], 0)
            masked = hit & (cnt > cap)
            cnt = np.where(masked, 0, cnt)
            st = np.where(cnt > 0, self.starts[pos], 0)
            nk[s0 : s0 + len(codes)] = valid.sum(1)
            nmask[s0 : s0 + len(codes)] = masked.sum(1)
            rowtot = cnt.sum(1)
            i = 0
            while i < len(codes):  # blocks that bound the number of postings expanded at once
                j, acc = i, 0
                while j < len(codes) and (acc + rowtot[j] <= max_postings or j == i):
                    acc += rowtot[j]
                    j += 1
                self._block(st[i:j], cnt[i:j], s0 + i, shared, best)
                i = j
        return shared, nk, best, nmask

    def _block(self, st, cnt, base, shared, best) -> None:
        q, W = cnt.shape
        c = cnt.ravel()
        tot = int(c.sum())
        if tot == 0:
            return
        s = st.ravel()
        rows = np.repeat(np.arange(q, dtype=np.int64), W)
        pos = np.repeat(s - (np.cumsum(c) - c), c) + np.arange(tot, dtype=np.int64)
        refs = self.ids[pos].astype(np.int64)
        qs = np.repeat(rows, c)
        key = qs * self.n_ref + refs
        uk, cts = np.unique(key, return_counts=True)
        uq = uk // self.n_ref
        ur = uk % self.n_ref
        bounds = np.flatnonzero(np.r_[True, uq[1:] != uq[:-1]])
        mx = np.maximum.reduceat(cts, bounds)
        grp = np.repeat(np.arange(len(bounds)), np.diff(np.r_[bounds, len(uq)]))
        cand = np.flatnonzero(cts == mx[grp])
        first = cand[np.r_[True, grp[cand][1:] != grp[cand][:-1]]]
        qid = uq[bounds]
        shared[base + qid] = mx.astype(np.int32)
        best[base + qid] = ur[first]
