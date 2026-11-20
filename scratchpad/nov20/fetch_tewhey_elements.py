"""Download the Tewhey oligo SEQUENCES from ENCODE, instead of reconstructing them.

Reconstructing the oligos from hg38 does not work and cannot be made to work: the base at
``chr:pos`` agrees with ``ref_allele`` only 24% of the time (chance), and even with the
coordinate convention fixed, the offset of the variant inside the 200-mer is not
identifiable from the reference alone -- every offset yields a valid 200-mer containing the
variant.

It is also unnecessary. Each FunctionalCharacterizationExperiment carries an
``elements_references`` dataset whose files include a FASTA of the actual designed oligos,
a few megabytes each. That is the authoritative library design, so the oligo IDs in the
element-quantification tables join straight onto real sequence.

    python3 -I scratchpad/nov20/fetch_tewhey_elements.py --out data/external/tewhey_elements
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

API = "https://www.encodeproject.org"
SEARCH = (
    f"{API}/search/?type=FunctionalCharacterizationExperiment"
    "&lab.title=Ryan+Tewhey%2C+JAX&assay_title=MPRA&limit=all&format=json"
)
WANTED_FORMATS = {"fasta", "tsv", "bed", "txt"}


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    g = get_json(SEARCH)["@graph"]
    seen: dict[str, dict] = {}
    for i, exp in enumerate(g):
        d = get_json(f"{API}/experiments/{exp['accession']}/?format=json")
        cell = (d.get("biosample_ontology") or {}).get("term_name", "unknown")
        for ref in d.get("elements_references") or []:
            racc = ref.get("accession") if isinstance(ref, dict) else str(ref)
            r = (
                ref
                if isinstance(ref, dict) and ref.get("files")
                else get_json(f"{API}/references/{racc}/?format=json")
            )
            for f in r.get("files", []):
                fo = f if isinstance(f, dict) else get_json(f"{API}{f}?format=json")
                if fo.get("file_format") not in WANTED_FORMATS:
                    continue
                if fo.get("status") != "released":
                    continue
                acc = fo["accession"]
                rec = seen.setdefault(
                    acc,
                    {
                        "reference": racc,
                        "file": acc,
                        "format": fo.get("file_format"),
                        "output_type": fo.get("output_type"),
                        "bytes": fo.get("file_size"),
                        "url": API + fo["href"],
                        "experiments": [],
                        "cells": set(),
                    },
                )
                rec["experiments"].append(exp["accession"])
                rec["cells"].add(cell)
        if (i + 1) % 15 == 0:
            print(f"  scanned {i + 1}/{len(g)}", flush=True)

    for r in seen.values():
        r["cells"] = sorted(r["cells"])
    total = sum(r["bytes"] or 0 for r in seen.values())
    print(f"{len(seen)} distinct element-reference files, {total / 1e6:.1f} MB", flush=True)
    with open(os.path.join(a.out, "MANIFEST.json"), "w") as fh:
        json.dump({"source": SEARCH, "files": list(seen.values())}, fh, indent=1)

    for i, r in enumerate(seen.values()):
        ext = {"fasta": "fa"}.get(r["format"], r["format"])
        dest = os.path.join(a.out, f"{r['reference']}__{r['file']}.{ext}")
        if os.path.exists(dest) and os.path.getsize(dest) == r["bytes"]:
            continue
        urllib.request.urlretrieve(r["url"], dest)
        print(
            f"  {i + 1}/{len(seen)} {os.path.basename(dest)} "
            f"({r['output_type']}, cells={r['cells']})",
            flush=True,
        )
    print(f"done -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
