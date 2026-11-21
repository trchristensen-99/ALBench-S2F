"""Download one HMEC MPRA fastq from the manifest, by index, resumably.

One file per array task: the 20 files are 107 GB in total and a single serial wget would
hold a node for hours, while ENCODE serves them independently. A file that is already
present at its manifest size is skipped, so a re-queued task after a node failure costs
nothing.

    python3 -I fetch_hmec_fastq.py --manifest <dir>/MANIFEST.json --out <dir>/fastq --index 0
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request


def entries(manifest: str) -> list[dict]:
    with open(manifest) as fh:
        man = json.load(fh)
    out = []
    for kind in ("rna", "dna"):
        for f in man[kind]["fastqs"]:
            out.append({**f, "kind": kind, "experiment": man[kind]["accession"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    rows = entries(a.manifest)
    if a.list or a.index < 0:
        for i, r in enumerate(rows):
            print(
                "%2d  %-4s rep%s R%s  %-12s %6.2f GB"
                % (i, r["kind"], r["replicate"], r["read"], r["accession"], (r["bytes"] or 0) / 1e9)
            )
        print("\n%d files, %.1f GB" % (len(rows), sum(r["bytes"] or 0 for r in rows) / 1e9))
        return

    r = rows[a.index]
    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(
        a.out, "%s_rep%s_R%s_%s.fastq.gz" % (r["kind"], r["replicate"], r["read"], r["accession"])
    )
    want = r["bytes"]
    if os.path.exists(dest) and (want is None or os.path.getsize(dest) == want):
        print("already complete: %s" % dest, flush=True)
        return
    tmp = dest + ".part"
    print("fetching %s -> %s (%.2f GB)" % (r["url"], dest, (want or 0) / 1e9), flush=True)
    urllib.request.urlretrieve(r["url"], tmp)
    got = os.path.getsize(tmp)
    if want is not None and got != want:
        raise SystemExit("size mismatch for %s: got %d, manifest says %d" % (dest, got, want))
    os.replace(tmp, dest)
    print("done %s (%d bytes)" % (dest, got), flush=True)


if __name__ == "__main__":
    main()
