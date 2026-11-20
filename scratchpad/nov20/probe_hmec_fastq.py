"""Can the primary-cell (HMEC) MPRA reads be assigned to oligos at all?

ENCODE released reads only for the HMEC and HEK293 experiments: no element
quantifications, no DNA/plasmid counts, and no barcode-to-oligo association file. Volume is
not the obstacle -- 59 GB and at most 44,486 oligos -- assignability is. Either the read
covers enough of the oligo to be matched against the design FASTA directly, or it carries
only a barcode, in which case the association map has to come from the lab and nothing can
be processed without it.

This settles it cheaply: an HTTP range request pulls the first slice of one fastq, and the
reads are matched against the element FASTA by exact k-mer lookup in both orientations.
A high hit rate means the oligo is in the read; a near-zero rate means barcodes only.

    python3 -I scratchpad/nov20/probe_hmec_fastq.py --out /tmp/hmec_probe --mb 120
"""

from __future__ import annotations

import argparse
import collections
import gzip
import os
import urllib.request

API = "https://www.encodeproject.org"
FASTQ = "ENCFF913CIO"  # HMEC, biorep 1, read 1
ELEMENTS = ("ENCSR091QEV__ENCFF770UJN.fa", "ENCSR935CNL__ENCFF090JTW.fa")
K = 30
_COMP = str.maketrans("ACGTN", "TGCAN")


def rc(s: str) -> str:
    return s.translate(_COMP)[::-1]


def read_fasta(path: str):
    with open(path, "rb") as probe:
        gz = probe.read(2) == b"\x1f\x8b"
    fh = gzip.open(path, "rt") if gz else open(path)
    name, buf = None, []
    with fh:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mb", type=int, default=120)
    ap.add_argument("--elements-dir", default="data/external/tewhey_elements")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dest = os.path.join(a.out, f"{FASTQ}.head.fastq.gz")
    if not os.path.exists(dest):
        url = f"{API}/files/{FASTQ}/@@download/{FASTQ}.fastq.gz"
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{a.mb * 1024 * 1024 - 1}"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as fh:
            fh.write(r.read())
    print(f"{dest}: {os.path.getsize(dest) / 1e6:.1f} MB", flush=True)

    # k-mer index over the designed oligos, both orientations
    index: dict[str, str] = {}
    n_oligo = 0
    for fn in ELEMENTS:
        p = os.path.join(a.elements_dir, fn)
        if not os.path.exists(p):
            print(f"  (missing {fn})")
            continue
        for name, s in read_fasta(p):
            n_oligo += 1
            s = s.upper()
            for seq in (s, rc(s)):
                for off in (0, max(0, len(seq) // 2 - K // 2), max(0, len(seq) - K)):
                    index.setdefault(seq[off : off + K], name)
    print(f"{n_oligo:,} design oligos -> {len(index):,} anchor {K}-mers", flush=True)

    reads = hits = 0
    per_oligo: collections.Counter = collections.Counter()
    lens: collections.Counter = collections.Counter()
    with gzip.open(dest, "rt", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1:
                continue
            s = line.strip().upper()
            if len(s) < K:
                continue
            reads += 1
            lens[len(s)] += 1
            found = None
            for off in range(0, len(s) - K + 1, 5):
                hit = index.get(s[off : off + K])
                if hit:
                    found = hit
                    break
            if found:
                hits += 1
                per_oligo[found] += 1
            if reads >= 200000:
                break

    print(f"\nreads examined: {reads:,}")
    print(f"read lengths  : {lens.most_common(3)}")
    print(f"matched an oligo by a {K}-mer: {hits:,}  ({hits / max(1, reads) * 100:.2f}%)")
    print(f"distinct oligos seen: {len(per_oligo):,}")
    if per_oligo:
        print("top oligos:", per_oligo.most_common(5))
    verdict = (
        "OLIGO IS IN THE READ -- processable without an association map"
        if hits / max(1, reads) > 0.2
        else "BARCODE ONLY -- needs the barcode-to-oligo association from the lab"
    )
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
